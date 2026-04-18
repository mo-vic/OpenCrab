"""Prompt templates for rollout analysis."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def get_analyzer_system_prompt() -> str:
    """Load the analyzer system prompt template.

    Returns:
        The system prompt for trajectory analysis as a string.
    """
    prompt_path = PROMPTS_DIR / "analyzer_system_prompt.txt"
    if prompt_path.exists():
        return prompt_path.read_text()
    # Fallback to inline prompt if file not found
    return (
        "You are an expert at analyzing AI conversation trajectories. "
        "Respond with ONLY valid JSON in this exact format, no other text:\n"
        '{"corrected_messages": [{"role": "user|assistant|system", "content": "..."}], '
        '"mistake_type": "factual|self_correction|tool_construction|domain_knowledge|reasoning_error|null", '
        '"summary": "brief summary of what was corrected"}'
    )
