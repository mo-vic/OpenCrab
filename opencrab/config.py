"""Configuration loading for OpenCrab.

Loads config from TOML files with environment variable overrides.
Environment variables take precedence over TOML values.

Convention: OPENCRAB_<SECTION>_<KEY> overrides config[SECTION][KEY].
For nested sections: OPENCRAB_SECTION_SUBSECTION_KEY.

Examples:
  OPENCRAB_INTERCEPT_PORT=9000 overrides [intercept].port
  OPENCRAB_SERVING_SGLANG_MODEL_PATH=/path/to/model overrides [serving.sglang].model_path
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore

# Global config cache
_config: dict[str, Any] = {}
_config_loaded: bool = False
_config_file: Path | None = None


def _get_env_override(section: str, key: str, value: Any) -> Any:
    """Check for environment variable override.

    Environment variable format: OPENCRAB_<SECTION>_<KEY>
    For nested sections, combine: OPENCRAB_SERVING_SGLANG_MODEL_PATH
    """
    env_key = f"OPENCRAB_{section.upper()}_{key.upper()}"
    env_val = os.environ.get(env_key)
    if env_val is not None:
        # Try to convert to appropriate type
        if isinstance(value, bool):
            return env_val.lower() in ("true", "1", "yes")
        elif isinstance(value, int):
            try:
                return int(env_val)
            except ValueError:
                return value
        elif isinstance(value, float):
            try:
                return float(env_val)
            except ValueError:
                return value
        return env_val
    return value


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: dict[str, Any], section: str = "") -> dict[str, Any]:
    """Recursively apply environment variable overrides to config."""
    if not isinstance(config, dict):
        return config

    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _apply_env_overrides(value, f"{section}_{key}" if section else key)
        else:
            result[key] = _get_env_override(section, key, value)
    return result


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from TOML file.

    Args:
        config_path: Path to config TOML file. If None, searches in order:
            - opencrab.toml in current directory
            - pyproject.toml in current directory (OpenCrab section)

    Returns:
        Configuration dictionary.

    Raises:
        RuntimeError: If TOML parsing fails or no config file found when required.
    """
    global _config, _config_loaded, _config_file

    if _config_loaded and _config_file == config_path:
        return _config

    config_path = config_path or _find_config_file()

    if config_path is None:
        # Return empty config, rely on defaults and env vars
        _config = {}
    elif not Path(config_path).exists():
        raise RuntimeError(f"Config file not found: {config_path}")
    else:
        with open(config_path, "rb") as f:
            if tomllib is None:
                raise RuntimeError(
                    "TOML parsing requires Python 3.11+ or the 'tomli' package. "
                    "Install with: pip install tomli"
                )
            raw_config = tomllib.load(f)

        # Extract [tool.opencrab] from pyproject.toml if present
        if config_path.name == "pyproject.toml":
            _config = raw_config.get("tool", {}).get("opencrab", raw_config)
        else:
            _config = raw_config

    # Apply environment variable overrides
    _config = _apply_env_overrides(_config)
    _config_loaded = True
    _config_file = config_path

    return _config


def _find_config_file() -> Path | None:
    """Search for config file in standard locations."""
    search_paths = [
        Path("opencrab.toml"),
        Path("pyproject.toml"),
        Path.cwd() / "config" / "opencrab.toml",
    ]

    for path in search_paths:
        if path.exists():
            return path
    return None


def get(section: str, key: str, default: Any = None) -> Any:
    """Get a configuration value.

    Args:
        section: Top-level config section (e.g., 'intercept', 'serving').
        key: Configuration key within that section.
        default: Default value if not found.

    Returns:
        Configuration value or default.
    """
    config = load_config()
    section_config = config.get(section, {})
    if isinstance(section_config, dict):
        return section_config.get(key, default)
    return default


def get_nested(section: str, *keys: str, default: Any = None) -> Any:
    """Get a nested configuration value.

    Args:
        section: Top-level config section.
        *keys: Sequence of nested keys.
        default: Default value if not found.

    Returns:
        Configuration value or default.
    """
    config = load_config()
    section_config = config.get(section, {})
    if not isinstance(section_config, dict):
        return default

    for key in keys:
        if isinstance(section_config, dict):
            section_config = section_config.get(key, {})
        else:
            return default
    return section_config if section_config else default


def reload() -> dict[str, Any]:
    """Force reload of configuration."""
    global _config_loaded, _config_file
    _config_loaded = False
    return load_config(_config_file)


# Convenience functions for common config lookups
def intercept_config() -> dict[str, Any]:
    """Get intercept layer configuration."""
    return load_config().get("intercept", {})


def serving_config() -> dict[str, Any]:
    """Get serving layer configuration."""
    return load_config().get("serving", {})


def rollout_config() -> dict[str, Any]:
    """Get rollout layer configuration."""
    return load_config().get("rollout", {})


def training_config() -> dict[str, Any]:
    """Get training layer configuration."""
    return load_config().get("training", {})
