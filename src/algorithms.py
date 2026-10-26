"""Utilities to expose available tsseg detectors with metadata for the app UI."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import logging
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
    complexity: str = ""  # Big-O complexity string (math, no $)

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
# Fallback used only when the algorithm isn't found in about_data.ALGO_BY_NAME.
_CATEGORY_MAP = {
    "vsax": "Subsequence-based",
    "changefinder": "Distribution-based",
    "snlds": "Decomposed",
    "patss": "Subsequence-based",
    "vqtss": "Decomposed",
    "random": "Baseline",
}


def _infer_category(name: str) -> str:
    """Infer algorithm family.

    Primary source is ``src.about_data.ALGO_BY_NAME`` (curated families).
    Falls back to ``_CATEGORY_MAP`` for detectors not listed there.
    """
    short = _strip_detector_suffix(name)
    # Local import to avoid a circular dependency at module load time.
    try:
        from .about_data import get_algo_entry

        entry = get_algo_entry(short)
        if entry is not None:
            return entry.family
    except Exception:
        pass
    name_lower = short.lower()
    for key, cat in _CATEGORY_MAP.items():
        if key in name_lower:
            return cat
    return "Other"


# Detectors that cannot run in the app's deployment environment.
_EXCLUDED_DETECTORS: set[str] = {
    "PatssDetector",  # pattern mining – missing deps
    "VQTSSDetector",  # deep learning – missing deps
    "TSCP2Detector",  # requires tensorflow not shipped on HF
    "SNLDSDetector",  # tensorflow + experimental
}


def _strip_detector_suffix(name: str) -> str:
    """Remove trailing 'Detector' from a class name for cleaner display."""
    if name.endswith("Detector"):
        return name[: -len("Detector")]
    return name


def _iter_exported_detectors_from_package() -> Iterable[Tuple[str, type]]:
    """Iterate detectors through ``tsseg.algorithms.__all__``.

    This path can fail when optional dependencies are missing because importing
    ``tsseg.algorithms`` eagerly imports many detector modules.
    """
    from tsseg import algorithms as algorithms_pkg

    exported = getattr(algorithms_pkg, "__all__", [])
    for name in exported:
        if name in _EXCLUDED_DETECTORS:
            continue
        try:
            obj = getattr(algorithms_pkg, name)
        except Exception:
            continue
        if inspect.isclass(obj) and hasattr(obj, "_tags"):
            yield name, obj


def _parse_algorithms_init_exports(init_file: Path) -> list[tuple[str, str]]:
    """Parse eager and lazy exports from ``tsseg.algorithms`` without importing it."""
    tree = ast.parse(init_file.read_text(encoding="utf-8"))
    modules: dict[str, str] = {}
    exported: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                modules[alias.asname or alias.name] = node.module
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "_LAZY_IMPORTS":
                    lazy = ast.literal_eval(node.value)
                    for name, (module, _attribute) in lazy.items():
                        modules[name] = module.lstrip(".")
                elif target.id == "__all__":
                    exported = ast.literal_eval(node.value)
    return [(modules[name], name) for name in exported if name in modules]


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
        if class_name in seen or class_name in _EXCLUDED_DETECTORS:
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
    from .about_data import get_complexity

    results: Dict[str, DetectorInfo] = {}
    for name, cls in _iter_exported_detectors():
        if name in _EXCLUDED_DETECTORS:
            logger.debug("Excluding detector %s (in _EXCLUDED_DETECTORS)", name)
            continue
        tags = getattr(cls, "_tags", {}) or {}
        detector_type = tags.get("detector_type", "unknown")
        short_name = _strip_detector_suffix(name)
        complexity = get_complexity(short_name)
        info = DetectorInfo(
            name=short_name,
            cls=cls,
            detector_type=detector_type,
            tags=tags,
            category=_infer_category(name),
            complexity=complexity,
        )
        results[short_name] = info
    return results


def get_detectors_by_type(detector_type: DetectorType) -> Dict[str, DetectorInfo]:
    """Return detectors filtered by detector_type tag."""
    return {name: info for name, info in get_all_detectors().items() if info.detector_type == detector_type}


def get_constructor_signature(cls: type) -> inspect.Signature:
    """Return the constructor signature excluding ``self``, ``*args``, ``**kwargs``."""
    sig = inspect.signature(cls.__init__)
    params = [
        param
        for param in sig.parameters.values()
        if param.name != "self" and param.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    ]
    return inspect.Signature(params)


def instantiate_detector(cls: type, params: Dict[str, Any]):
    """Instantiate a detector with validated parameters."""
    try:
        return cls(**params)
    except TypeError as exc:
        raise ValueError(f"Failed to instantiate {cls.__name__}: {exc}") from exc
