"""Dynamic parameter rendering for Gradio based on detector introspection + YAML configs."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent / "config"
_YAML_CACHE: Dict[str, dict] = {}


def _load_yaml_config(detector_name: str) -> Optional[dict]:
    """Load YAML config with tunable parameter ranges for a detector."""
    if detector_name in _YAML_CACHE:
        return _YAML_CACHE[detector_name]

    # Try various name mappings
    candidates = [
        detector_name.lower().replace("detector", ""),
        detector_name.lower().replace("detector", "").replace("_", "-"),
        detector_name.lower(),
    ]

    for candidate in candidates:
        path = _CONFIG_DIR / f"{candidate}.yaml"
        if path.exists():
            try:
                with open(path) as f:
                    config = yaml.safe_load(f)
                _YAML_CACHE[detector_name] = config
                return config
            except Exception as e:
                logger.warning(f"Failed to load YAML config {path}: {e}")

    _YAML_CACHE[detector_name] = None
    return None


def get_tunable_ranges(detector_name: str) -> Dict[str, list]:
    """Get tunable parameter ranges from YAML config.

    Returns a dict like {"model": ["l1", "l2", "rbf"], "penalty": [5, 20, 100]}.
    """
    config = _load_yaml_config(detector_name)
    if config is None:
        return {}

    tunable = config.get("tunable_parameters", [])
    ranges = {}
    if isinstance(tunable, list):
        for item in tunable:
            if isinstance(item, dict):
                ranges.update(item)
    return ranges


def get_param_info(cls: type) -> list[dict]:
    """Extract parameter info from a detector constructor for UI rendering.

    Returns a list of dicts, each with:
        - name: parameter name
        - type: 'bool' | 'int' | 'float' | 'str' | 'choice' | 'text'
        - default: default value
        - choices: list of choices (if applicable)
        - min/max/step: numeric bounds (if applicable)
    """
    sig = inspect.signature(cls.__init__)
    detector_name = cls.__name__
    tunable_ranges = get_tunable_ranges(detector_name)

    params = []
    for param in sig.parameters.values():
        if param.name == "self":
            continue
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue

        default = param.default if param.default is not inspect._empty else None
        annotation = param.annotation if param.annotation is not inspect._empty else None

        # Skip internal / axis parameters
        if param.name in ("axis",):
            continue

        info = {
            "name": param.name,
            "default": default,
            "annotation": annotation,
        }

        # Check if this param has tunable choices from YAML
        if param.name in tunable_ranges:
            choices = tunable_ranges[param.name]
            if isinstance(choices, list) and len(choices) > 1:
                # If default is in choices, use choices mode
                if all(isinstance(c, str) for c in choices):
                    info["type"] = "choice"
                    info["choices"] = choices
                    if default not in choices:
                        choices.insert(0, default) if default is not None else None
                elif all(isinstance(c, (int, float)) for c in choices):
                    # For numeric tunable params, show as slider with range based on choices
                    if isinstance(default, bool):
                        info["type"] = "bool"
                    elif isinstance(default, int) and all(isinstance(c, int) for c in choices):
                        info["type"] = "int"
                        info["min"] = min(choices)
                        info["max"] = max(choices)
                        info["step"] = 1
                        info["choices"] = choices
                    else:
                        info["type"] = "float"
                        info["min"] = float(min(choices))
                        info["max"] = float(max(choices))
                        info["step"] = round((max(choices) - min(choices)) / 20, 6)
                        info["choices"] = choices
                else:
                    info["type"] = "choice"
                    info["choices"] = [str(c) for c in choices]
                params.append(info)
                continue

        # Fallback: infer type from default value
        if isinstance(default, bool):
            info["type"] = "bool"
        elif isinstance(default, int):
            info["type"] = "int"
            lower = 0 if default >= 0 else int(default * 3)
            upper = default * 3 + 10 if default >= 0 else -default * 3 + 10
            upper = max(upper, lower + 5)
            info["min"] = lower
            info["max"] = upper
            info["step"] = 1
        elif isinstance(default, float):
            info["type"] = "float"
            span = max(abs(default), 0.1)
            info["min"] = round(float(max(0, default - 3 * span)), 6)
            info["max"] = round(float(default + 3 * span), 6)
            info["step"] = round(span / 20, 6)
        elif isinstance(default, str):
            # Check annotation for Literal
            info["type"] = "str"
        elif default is None:
            info["type"] = "text"
        elif isinstance(default, (list, tuple, dict)):
            info["type"] = "text"
        else:
            info["type"] = "text"

        params.append(info)

    return params
