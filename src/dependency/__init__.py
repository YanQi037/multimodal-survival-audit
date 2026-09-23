"""Inference-time dependency and ranking-flip metrics."""

from .shuffle import (
    dependency_metrics,
    deterministic_patient_derangement,
    generated_simple_fusion_shuffle,
    provided_shuffle_audit,
    summarize_dependency,
)

__all__ = [
    "dependency_metrics",
    "deterministic_patient_derangement",
    "generated_simple_fusion_shuffle",
    "provided_shuffle_audit",
    "summarize_dependency",
]

