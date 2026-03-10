"""Utilities to expose available tsseg detectors with metadata for the app UI."""

from __future__ import annotations

import inspect
import importlib
import importlib.util
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

logger = logging.getLogger(__name__)

DetectorType = str

DETECTOR_CATEGORIES = {
    "change_point_detection": "Change Point Detection",
    "state_detection": "State Detection",
}


@dataclass
class DetectorInfo:
    """Metadata for a single detector class."""

    name: str
    cls: type
    detector_type: DetectorType
    tags: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    category: str = ""

    @property
    def dependencies(self) -> list[str]:
        """Return list of extra Python dependencies."""
        deps = self.tags.get("python_dependencies") or []
        if isinstance(deps, str):
            deps = [d.strip() for d in deps.split(",")]
        return deps

    @property
    def is_available(self) -> bool:
        """Check if all required dependencies are installed."""
        import importlib.util

        for dep in self.dependencies:
            if importlib.util.find_spec(dep) is None:
                return False
        return True

    @property
    def display_label(self) -> str:
        """Format detector name with dependency info for dropdown."""
        extras = self.dependencies
        suffix = ""
        if extras:
            suffix = f"  ({'·'.join(extras)})"
        task = "CPD" if self.detector_type == "change_point_detection" else "SD"
        return f"{self.name} [{task}]{suffix}"


# -- Detector category heuristics based on module path / name --
_CATEGORY_MAP = {
    "pelt": "Ruptures (offline)",
    "binseg": "Ruptures (offline)",
    "bottomup": "Ruptures (offline)",
    "dynp": "Ruptures (offline)",
    "window": "Ruptures (offline)",
    "kcpd": "Ruptures (offline)",
    "amoc": "Ruptures (offline)",
    "bocd": "Bayesian",
    "hdp_hsmm": "Bayesian",
    "hidalgo": "Bayesian",
    "clasp": "Matrix Profile",
    "clap": "Matrix Profile",
    "fluss": "Matrix Profile",
    "espresso": "Matrix Profile",
    "ggs": "Statistical",
    "eagglo": "Statistical",
    "igts": "Statistical",
    "icid": "Kernel / Isolation",
    "vsax": "Symbolic",
    "ticc": "Graphical Model",
    "tglad": "Deep Learning",
    "tire": "Deep Learning",
    "e2usd": "Deep Learning",
    "time2state": "Deep Learning",
    "tscp2": "Deep Learning",
    "vqtss": "Deep Learning",
    "prophet": "Forecasting",
    "autoplait": "Regime (HMM)",
    "patss": "Pattern Mining",
    "random": "Baseline",
}


def _infer_category(name: str) -> str:
    """Infer algorithm category from the class name."""
    name_lower = name.lower().replace("detector", "")
    for key, cat in _CATEGORY_MAP.items():
        if key in name_lower:
            return cat
    return "Other"


# Detectors excluded from the app (broken dependencies, unavailable on HF, etc.)
_EXCLUDED_DETECTORS: set[str] = {
    "PatssDetector",   # pattern mining – missing deps
    "VQTSSDetector",   # deep learning – missing deps
    "VSAXDetector",    # symbolic – broken on HF
    "AutoPlaitDetector",  # requires C binary not available on HF
    "TSCP2Detector",   # requires tensorflow not shipped on HF
}


def _iter_exported_detectors_from_package() -> Iterable[Tuple[str, type]]:
    """Iterate detectors through ``tsseg.algorithms.__all__``.

    This path can fail when optional dependencies are missing because importing
    ``tsseg.algorithms`` eagerly imports many detector modules.
    """
    from tsseg import algorithms as algorithms_pkg

    exported = getattr(algorithms_pkg, "__all__", [])
    for name in exported:
        try:
            obj = getattr(algorithms_pkg, name)
        except Exception:
            continue
        if inspect.isclass(obj) and hasattr(obj, "_tags"):
            yield name, obj


def _parse_algorithms_init_exports(init_file: Path) -> list[tuple[str, str]]:
    """Parse `tsseg/algorithms/__init__.py` to extract `(module, class)` exports."""
    text = init_file.read_text(encoding="utf-8")
    exports: list[tuple[str, str]] = []

    in_multiline_import = False
    multiline_module = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if in_multiline_import:
            if line.startswith(")"):
                in_multiline_import = False
                multiline_module = ""
                continue
            name = line.split("#", 1)[0].strip().rstrip(",")
            if name:
                exports.append((multiline_module, name))
            continue

        m = re.match(r"^from \.([a-zA-Z0-9_\.]+) import \($", line)
        if m:
            in_multiline_import = True
            multiline_module = m.group(1)
            continue

        m = re.match(r"^from \.([a-zA-Z0-9_\.]+) import ([A-Za-z0-9_]+)$", line)
        if m:
            exports.append((m.group(1), m.group(2)))

    return exports


def _iter_exported_detectors_from_source() -> Iterable[Tuple[str, type]]:
    """Iterate detectors by importing module-by-module with graceful fallback."""
    spec = importlib.util.find_spec("tsseg.algorithms")
    if spec is None or spec.origin is None:
        return

    init_file = Path(spec.origin)
    if not init_file.exists():
        return

    seen: set[str] = set()
    for module_rel, class_name in _parse_algorithms_init_exports(init_file):
        if class_name in seen:
            continue
        module_name = f"tsseg.algorithms.{module_rel}"
        try:
            module = importlib.import_module(module_name)
            obj = getattr(module, class_name)
        except Exception as exc:
            logger.debug("Skipping %s from %s: %s", class_name, module_name, exc)
            continue

        if inspect.isclass(obj) and hasattr(obj, "_tags"):
            seen.add(class_name)
            yield class_name, obj


def _iter_exported_detectors() -> Iterable[Tuple[str, type]]:
    """Iterate over exported detector classes with optional-deps tolerance."""
    try:
        yielded = False
        for item in _iter_exported_detectors_from_package():
            yielded = True
            yield item
        if yielded:
            return
    except Exception as exc:
        logger.warning(
            "Could not import tsseg.algorithms directly (%s). Falling back to resilient discovery.",
            exc,
        )

    yield from _iter_exported_detectors_from_source()


def get_all_detectors() -> Dict[str, DetectorInfo]:
    """Return all detectors with metadata (excluding broken/unavailable ones)."""
    results: Dict[str, DetectorInfo] = {}
    for name, cls in _iter_exported_detectors():
        if name in _EXCLUDED_DETECTORS:
            logger.debug("Excluding detector %s (in _EXCLUDED_DETECTORS)", name)
            continue
        tags = getattr(cls, "_tags", {}) or {}
        detector_type = tags.get("detector_type", "unknown")
        info = DetectorInfo(
            name=name,
            cls=cls,
            detector_type=detector_type,
            tags=tags,
            category=_infer_category(name),
        )
        results[name] = info
    return results


def get_detectors_by_type(detector_type: DetectorType) -> Dict[str, DetectorInfo]:
    """Return detectors filtered by detector_type tag."""
    return {
        name: info
        for name, info in get_all_detectors().items()
        if info.detector_type == detector_type
    }


def get_constructor_signature(cls: type) -> inspect.Signature:
    """Return the constructor signature excluding ``self``, ``*args``, ``**kwargs``."""
    sig = inspect.signature(cls.__init__)
    params = [
        param
        for param in sig.parameters.values()
        if param.name != "self"
        and param.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    ]
    return inspect.Signature(params)


def instantiate_detector(cls: type, params: Dict[str, Any]):
    """Instantiate a detector with validated parameters."""
    try:
        return cls(**params)
    except TypeError as exc:
        raise ValueError(f"Failed to instantiate {cls.__name__}: {exc}") from exc
