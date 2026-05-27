"""Dynamic parameter rendering for Gradio based on tsseg param_schema + YAML fallback.

Priority order for each detector:
1. ``tsseg.algorithms.param_schema.get_ui_hints`` (declarative schema with
   data-dependent bounds).
2. YAML tunable ranges in ``src/config/<detector>.yaml``.
3. Pure introspection of the ``__init__`` signature (legacy fallback).

When a *data context* is provided (``n_samples``, ``n_channels``), data-
dependent constraints are resolved and default values that violate them are
**automatically clamped** with a warning.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent / "config"
_YAML_CACHE: Dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Data context helper
# ---------------------------------------------------------------------------


def build_data_context(
    signal: Any,
    *,
    n_channels: int | None = None,
) -> Dict[str, Any]:
    """Build a ``data_ctx`` dict from a signal array.

    Parameters
    ----------
    signal : array-like
        The time-series array, shape ``(n_samples,)`` or ``(n_samples, n_dims)``.
    n_channels : int, optional
        Override channel count (inferred from *signal* otherwise).

    Returns
    -------
    dict
        ``{"n_samples": …, "n_channels": …}``
    """
    import numpy as np

    sig = np.asarray(signal)
    n_samples = sig.shape[0]
    if n_channels is None:
        n_channels = sig.shape[1] if sig.ndim > 1 else 1
    return {"n_samples": n_samples, "n_channels": n_channels}


# ---------------------------------------------------------------------------
# YAML config (unchanged, used as fallback / override)
# ---------------------------------------------------------------------------


def _load_yaml_config(detector_name: str) -> Optional[dict]:
    """Load YAML config with tunable parameter ranges for a detector."""
    if detector_name in _YAML_CACHE:
        return _YAML_CACHE[detector_name]

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
    """Get tunable parameter ranges from YAML config."""
    config = _load_yaml_config(detector_name)
    if config is None:
        return {}

    tunable = config.get("tunable_parameters", [])
    ranges: Dict[str, list] = {}
    if isinstance(tunable, list):
        for item in tunable:
            if isinstance(item, dict):
                ranges.update(item)
    return ranges


# ---------------------------------------------------------------------------
# Schema-aware parameter info
# ---------------------------------------------------------------------------


def _has_parameter_schema(cls: type) -> bool:
    """Return True if *cls* declares a non-empty ``_parameter_schema``."""
    schema = getattr(cls, "_parameter_schema", None)
    return bool(schema)


def _clamp_default(
    value: Any,
    lo: float | int | None,
    hi: float | int | None,
    ptype: str,
) -> Tuple[Any, bool]:
    """Clamp *value* to ``[lo, hi]``.  Returns ``(clamped, was_adjusted)``."""
    if value is None:
        return value, False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value, False

    adjusted = False
    if lo is not None and v < lo:
        v = lo
        adjusted = True
    if hi is not None and v > hi:
        v = hi
        adjusted = True

    if adjusted:
        v = int(v) if ptype == "int" else v
    return v, adjusted


def get_param_info(
    cls: type,
    data_ctx: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Extract parameter info for UI rendering.

    When a ``_parameter_schema`` is declared on *cls*, it is preferred over
    the legacy YAML / introspection path.  In both cases, YAML tunable
    ranges are merged in to refine the choices.

    Parameters
    ----------
    cls : type
        Detector class.
    data_ctx : dict, optional
        Data context (``{"n_samples": …, "n_channels": …}``).

    Returns
    -------
    list[dict]
        Per-parameter dicts with keys: ``name``, ``type``, ``default``,
        ``min``, ``max``, ``step``, ``choices``, ``description``, ``group``,
        ``nullable``, ``hidden``, ``adjusted``, ``adjust_reason``.
    """
    if _has_parameter_schema(cls):
        return _param_info_from_schema(cls, data_ctx)
    return _param_info_legacy(cls)


# ---------------------------------------------------------------------------
# Schema-based path (new)
# ---------------------------------------------------------------------------


def _param_info_from_schema(
    cls: type,
    data_ctx: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Build param info list from ``get_ui_hints``."""
    from tsseg.algorithms.param_schema import get_ui_hints

    hints = get_ui_hints(cls, data_ctx=data_ctx)
    if not hints:
        return _param_info_legacy(cls)

    detector_name = cls.__name__
    tunable_ranges = get_tunable_ranges(detector_name)

    params: List[Dict[str, Any]] = []
    for pname, h in hints.items():
        if h.get("hidden", False):
            continue

        ptype = h.get("type", "any")
        default = h.get("default")
        lo = h.get("min")
        hi = h.get("max")
        nullable = h.get("nullable", False)
        choices = h.get("choices")
        description = h.get("description", "")
        group = h.get("group", "")

        info: Dict[str, Any] = {
            "name": pname,
            "default": default,
            "description": description,
            "group": group,
            "nullable": nullable,
            "adjusted": False,
            "adjust_reason": "",
        }

        # --- Merge YAML tunable ranges if present --------------------------
        if pname in tunable_ranges and choices is None:
            yaml_choices = tunable_ranges[pname]
            if isinstance(yaml_choices, list) and all(isinstance(c, str) for c in yaml_choices):
                choices = yaml_choices

        # --- Determine UI widget type --------------------------------------
        if choices:
            info["type"] = "choice"
            info["choices"] = choices
            if default is not None and default not in choices:
                choices.insert(0, default)
        elif ptype == "bool":
            info["type"] = "bool"
        elif ptype == "int":
            info["type"] = "int"
            effective_lo = lo if lo is not None else 0
            effective_hi = hi if hi is not None else _fallback_upper(default, effective_lo)
            info["min"] = int(effective_lo)
            info["max"] = int(effective_hi)
            info["step"] = 1
            # Auto-clamp default
            if default is not None:
                clamped, was = _clamp_default(default, effective_lo, effective_hi, "int")
                if was:
                    info["default"] = clamped
                    info["adjusted"] = True
                    info["adjust_reason"] = (
                        f"Default {pname}={default} adjusted to {clamped} (series of {data_ctx['n_samples']} points)"
                        if data_ctx
                        else f"Default {pname}={default} out of bounds, adjusted to {clamped}"
                    )
        elif ptype == "float":
            info["type"] = "float"
            effective_lo = lo if lo is not None else 0.0
            effective_hi = hi if hi is not None else _fallback_upper_f(default, effective_lo)
            info["min"] = float(effective_lo)
            info["max"] = float(effective_hi)
            info["step"] = round((effective_hi - effective_lo) / 100, 6) or 0.01
            # Auto-clamp default
            if default is not None:
                clamped, was = _clamp_default(default, effective_lo, effective_hi, "float")
                if was:
                    info["default"] = clamped
                    info["adjusted"] = True
                    info["adjust_reason"] = (
                        f"Default {pname}={default} adjusted to {clamped} (series of {data_ctx['n_samples']} points)"
                        if data_ctx
                        else f"Default {pname}={default} out of bounds, adjusted to {clamped}"
                    )
        elif ptype == "str":
            info["type"] = "str"
        else:
            info["type"] = "text"

        params.append(info)

    return params


def _fallback_upper(default: Any, lo: float | int) -> int:
    """Heuristic upper bound when the schema has no finite upper bound."""
    if default is not None:
        try:
            d = int(default)
            return max(d * 3 + 10, int(lo) + 10)
        except (TypeError, ValueError):
            pass
    return int(lo) + 100


def _fallback_upper_f(default: Any, lo: float) -> float:
    """Heuristic upper bound for floats."""
    if default is not None:
        try:
            d = float(default)
            span = max(abs(d), 0.1)
            return round(d + 3 * span, 6)
        except (TypeError, ValueError):
            pass
    return lo + 10.0


# ---------------------------------------------------------------------------
# Legacy path (introspection + YAML, no param_schema)
# ---------------------------------------------------------------------------


def _param_info_legacy(cls: type) -> List[Dict[str, Any]]:
    """Original introspection-based param extraction (unchanged logic)."""
    sig = inspect.signature(cls.__init__)
    detector_name = cls.__name__
    tunable_ranges = get_tunable_ranges(detector_name)

    params: List[Dict[str, Any]] = []
    for param in sig.parameters.values():
        if param.name == "self":
            continue
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue

        default = param.default if param.default is not inspect._empty else None
        annotation = param.annotation if param.annotation is not inspect._empty else None

        if param.name in ("axis",):
            continue

        info: Dict[str, Any] = {
            "name": param.name,
            "default": default,
            "annotation": annotation,
            "description": "",
            "group": "",
            "nullable": False,
            "adjusted": False,
            "adjust_reason": "",
        }

        # YAML tunable choices
        if param.name in tunable_ranges:
            choices = tunable_ranges[param.name]
            if isinstance(choices, list) and len(choices) > 1:
                if all(isinstance(c, str) for c in choices):
                    info["type"] = "choice"
                    info["choices"] = choices
                    if default not in choices:
                        choices.insert(0, default) if default is not None else None
                elif all(isinstance(c, (int, float)) for c in choices):
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

        # Fallback: infer from default
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
            info["type"] = "str"
        elif default is None:
            info["type"] = "text"
        elif isinstance(default, (list, tuple, dict)):
            info["type"] = "text"
        else:
            info["type"] = "text"

        params.append(info)

    return params


# ---------------------------------------------------------------------------
# Validation & adjustment
# ---------------------------------------------------------------------------


def validate_params_for_detector(
    cls: type,
    params: Dict[str, Any],
    data_ctx: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Validate and auto-adjust parameters before running a detector.

    Parameters
    ----------
    cls : type
        Detector class.
    params : dict
        User-supplied parameter values.
    data_ctx : dict, optional
        Data context for data-dependent constraints.

    Returns
    -------
    adjusted_params : dict
        Parameters with out-of-bounds values clamped.
    warnings : list[str]
        Human-readable warnings about auto-adjusted values.
    errors : list[str]
        Hard errors that cannot be auto-fixed.
    """
    if not _has_parameter_schema(cls):
        return params, [], []

    from tsseg.algorithms.param_schema import get_ui_hints
    from tsseg.algorithms.param_schema import validate_params as _validate

    hints = get_ui_hints(cls, data_ctx=data_ctx)
    adjusted = dict(params)
    warnings: List[str] = []
    errors: List[str] = []

    # --- Auto-clamp numeric values to effective bounds --------------------
    for pname, h in hints.items():
        if pname not in adjusted:
            continue
        val = adjusted[pname]
        if val is None:
            continue

        lo = h.get("min")
        hi = h.get("max")
        ptype = h.get("type", "any")

        if ptype in ("int", "float") and (lo is not None or hi is not None):
            clamped, was = _clamp_default(val, lo, hi, ptype)
            if was:
                adjusted[pname] = clamped
                if data_ctx:
                    warnings.append(
                        f"**{pname}** = {val} → {clamped} (series: {data_ctx.get('n_samples', '?')} points)"
                    )
                else:
                    warnings.append(f"**{pname}** = {val} → {clamped} (out of bounds)")

    # --- Run the full schema validator for cross-constraints ---------------
    try:
        # Build a temporary instance to use validate_params
        instance = cls.__new__(cls)
        # Set params so get_params works (sklearn interface)
        for k, v in adjusted.items():
            setattr(instance, k, v)
        errs = _validate(instance, data_ctx=data_ctx)
        for e in errs:
            # If the error is about a value we already clamped, downgrade to warning
            errors.append(e)
    except Exception as exc:
        logger.debug(f"Schema validation skipped: {exc}")

    return adjusted, warnings, errors
