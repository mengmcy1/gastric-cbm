"""Adaptive3DGS core interfaces."""

from .clb import Camera, CanonicalLayerBundle, CanonicalLayerPatch, Provenance, SupportType
from .evaluation import GeometryPrediction, GeometryTarget, evaluate_occlusion_geometry
from .io import load_bundle, save_bundle
from .providers import OcclusionHiddenProvider, OutsideFOVProvider, ProviderContext, ProviderResult
from .registry import ProviderRegistry
from .supervision import HiddenLayerEvidence, TargetRGBDObservation, build_source_hidden_evidence
from .validation import ValidationError, ValidationReport, validate_patch, validate_result

__all__ = [
    "CanonicalLayerPatch",
    "CanonicalLayerBundle",
    "Camera",
    "GeometryPrediction",
    "GeometryTarget",
    "HiddenLayerEvidence",
    "OcclusionHiddenProvider",
    "OutsideFOVProvider",
    "Provenance",
    "ProviderContext",
    "ProviderRegistry",
    "ProviderResult",
    "SupportType",
    "TargetRGBDObservation",
    "ValidationError",
    "ValidationReport",
    "validate_patch",
    "validate_result",
    "load_bundle",
    "save_bundle",
    "evaluate_occlusion_geometry",
    "build_source_hidden_evidence",
]
