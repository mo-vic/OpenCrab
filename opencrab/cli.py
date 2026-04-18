"""CLI entry point for OpenCrab."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import structlog
import typer

from opencrab.intercept import app as intercept_app
from opencrab.rollout import (
    JSONLTransform,
    TrainingSampleExtractor,
    TrajectoryAnalyzer,
)
from opencrab.serving import app as serving_app
from opencrab.training import TrainingConfig, generate_model_card, get_pipeline

app = typer.Typer(help="OpenCrab — Self-distilling learning framework")
training_app = typer.Typer(help="Training commands")
app.add_typer(training_app, name="training")

log = structlog.get_logger()

# Persistent job tracking via file
_JOBS_FILE = Path("./.opencrab_jobs.json")


def _load_jobs() -> dict[str, dict[str, Any]]:
    """Load jobs from persistent storage."""
    if _JOBS_FILE.exists():
        try:
            with open(_JOBS_FILE) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_jobs(jobs: dict[str, dict[str, Any]]) -> None:
    """Save jobs to persistent storage."""
    with open(_JOBS_FILE, "w") as f:
        json.dump(jobs, f)


def _create_job(method: str, data_path: str, output_dir: str) -> str:
    """Create a new training job."""
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    jobs = _load_jobs()
    jobs[job_id] = {
        "status": "pending",
        "method": method,
        "data_path": str(data_path),
        "output_dir": str(output_dir),
    }
    _save_jobs(jobs)
    return job_id


def _get_job(job_id: str) -> dict[str, Any] | None:
    """Get job by ID."""
    jobs = _load_jobs()
    return jobs.get(job_id)


def _update_job(job_id: str, updates: dict[str, Any]) -> None:
    """Update job status."""
    jobs = _load_jobs()
    if job_id in jobs:
        jobs[job_id].update(updates)
        _save_jobs(jobs)


@app.command()
def start(  # noqa: N802
    intercept_host: str = "0.0.0.0",
    intercept_port: int = 8080,
    serving_host: str = "0.0.0.0",
    serving_port: int = 8081,
    openai_key: str | None = None,
    anthropic_key: str | None = None,
    serving_url: str = "http://localhost:8081",
    distilled_url: str = "http://localhost:8000/v1/chat/completions",
    classifier: str = "self_classifier",
) -> None:
    """Start OpenCrab (intercept proxy + serving server).

    By default starts both intercept and serving servers. Use --intercept-only
    to start only the intercept proxy.
    """
    import threading

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    os.environ["OPENCRAB_SERVING_URL"] = serving_url
    os.environ["DISTILLED_MODEL_URL"] = distilled_url
    os.environ["ROUTER_CLASSIFIER"] = classifier

    import uvicorn

    # Start serving server in background thread
    serving_thread = threading.Thread(
        target=lambda: uvicorn.run(serving_app, host=serving_host, port=serving_port),
        daemon=True,
    )
    serving_thread.start()
    log.info("serving_server_started", host=serving_host, port=serving_port)

    # Start intercept server (blocking)
    log.info("intercept_server_started", host=intercept_host, port=intercept_port)
    uvicorn.run(intercept_app, host=intercept_host, port=intercept_port)


@app.command()
def startIntercept(  # noqa: N802
    host: str = "0.0.0.0",
    port: int = 8080,
    openai_key: str | None = None,
    anthropic_key: str | None = None,
    serving_url: str = "http://localhost:8081",
) -> None:
    """Start the intercept proxy server (standalone).

    This starts only the intercept proxy without the serving layer.
    Use 'start' to run both servers together.
    """
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    os.environ["OPENCRAB_SERVING_URL"] = serving_url

    import uvicorn

    log.info("starting_intercept_server", host=host, port=port, serving_url=serving_url)
    uvicorn.run(intercept_app, host=host, port=port)


@app.command()
def startServing(  # noqa: N802
    host: str = "0.0.0.0",
    port: int = 8081,
    distilled_url: str = "http://localhost:8000/v1/chat/completions",
    classifier: str = "self_classifier",
) -> None:
    """Start the serving server (standalone).

    This starts only the model serving layer without the intercept proxy.
    Use 'start' to run both servers together.
    """
    os.environ["DISTILLED_MODEL_URL"] = distilled_url
    os.environ["ROUTER_CLASSIFIER"] = classifier

    import uvicorn

    log.info(
        "starting_serving_server",
        host=host,
        port=port,
        model_url=distilled_url,
        classifier=classifier,
    )
    uvicorn.run(serving_app, host=host, port=port)


@app.command()
async def analyze(
    trajectory_path: Path,
    output_path: Path | None = None,
    analyzer_key: str | None = None,
) -> None:
    """Analyze trajectories for mistakes."""
    import json

    if analyzer_key:
        os.environ["ANTHROPIC_API_KEY"] = analyzer_key

    analyzer = TrajectoryAnalyzer()
    extractor = TrainingSampleExtractor()

    with open(trajectory_path) as f:
        trajectories = [json.loads(line) for line in f]

    all_samples = []
    for traj in trajectories:
        corrected = await analyzer.analyze(traj)
        samples = extractor.extract(traj, corrected)
        all_samples.extend(samples)

    if output_path:
        transform = JSONLTransform(output_path)
        await transform.write_samples(all_samples)
        log.info("analysis_complete", samples=len(all_samples), output=str(output_path))
    else:
        for sample in all_samples:
            print(sample.to_training_format())


@app.command()
def train(
    data_path: Path,
    method: str = "lora",
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: Path = Path("./model_output"),
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
) -> None:
    """Fine-tune the distilled model."""
    job_id = _create_job(method, str(data_path), str(output_dir))
    _update_job(job_id, {"status": "running"})

    config = TrainingConfig(
        base_model=base_model,
        method=method,
        output_dir=str(output_dir),
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    pipeline = get_pipeline(method)

    import asyncio

    try:

        async def run_training():
            await pipeline.train(config, data_path)
            await pipeline.save(output_dir)

        asyncio.run(run_training())

        # Compute sample count from data file
        sample_count = 0
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    sample_count += 1

        # Get metrics from pipeline if available
        final_loss = getattr(pipeline, "final_loss", None)
        training_duration = getattr(pipeline, "training_duration_minutes", None)

        generate_model_card(config, output_dir, sample_count, final_loss, training_duration)
        _update_job(job_id, {"status": "completed"})
        log.info("training_complete", job_id=job_id, output=str(output_dir), samples=sample_count)
    except Exception as e:
        _update_job(job_id, {"status": "failed", "error": str(e)})
        log.error("training_failed", job_id=job_id, error=str(e))
        raise


@training_app.command("status")
def training_status(job_id: str) -> None:
    """Check training job status."""
    job = _get_job(job_id)
    if not job:
        typer.echo(f"Job {job_id} not found")
        raise typer.Exit(1)

    status = job.get("status", "unknown")
    typer.echo(f"Job ID: {job_id}")
    typer.echo(f"Status: {status}")
    if "error" in job:
        typer.echo(f"Error: {job['error']}")


@training_app.command("cancel")
def training_cancel(job_id: str) -> None:
    """Cancel a training job."""
    job = _get_job(job_id)
    if not job:
        typer.echo(f"Job {job_id} not found")
        raise typer.Exit(1)

    if job.get("status") in ("completed", "failed", "cancelled"):
        typer.echo(f"Job {job_id} is already {job['status']}")
        raise typer.Exit(1)

    _update_job(job_id, {"status": "cancelled"})
    typer.echo(f"Job {job_id} cancelled")


@training_app.command("list-models")
def training_list_models(output_dir: Path = Path("./model_output")) -> None:
    """List trained models."""
    model_card_path = output_dir / "model_card.json"
    if model_card_path.exists():
        with open(model_card_path) as f:
            card = json.load(f)
        typer.echo(f"Model ID: {card.get('model_id', 'unknown')}")
        typer.echo(f"Base Model: {card.get('base_model', 'unknown')}")
        typer.echo(f"Training Method: {card.get('training_method', 'unknown')}")
        typer.echo(f"Training Date: {card.get('training_date', 'unknown')}")
    else:
        typer.echo("No trained model found in output directory")


@app.command()
def version() -> None:
    """Show OpenCrab version."""
    from opencrab import __version__

    typer.echo(f"OpenCrab {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
