"""Utilities to expose available tsseg detectors with metadata for the app UI."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Tuple

from tsseg import algorithms as algorithms_pkg

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


def _iter_exported_detectors() -> Iterable[Tuple[str, type]]:
    """Iterate over all exported detector classes from tsseg.algorithms."""
    exported = getattr(algorithms_pkg, "__all__", [])
    for name in exported:
        try:
            obj = getattr(algorithms_pkg, name)
        except AttributeError:
            continue
        if inspect.isclass(obj) and hasattr(obj, "_tags"):
            yield name, obj


def get_all_detectors() -> Dict[str, DetectorInfo]:
    """Return all detectors with metadata."""
    results: Dict[str, DetectorInfo] = {}
    for name, cls in _iter_exported_detectors():
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
