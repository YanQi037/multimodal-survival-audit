"""Global pathology/molecular weight sweep using pooled fold-local pairs."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from src.metrics.pairs import (
    GROUP_M,
    GROUP_P,
    build_fold_pairs,
    evaluate_vectors,
    vectors_from_column,
)


def run_weight_sweep(
    frame: pd.DataFrame,
    weights: Iterable[float] | None = None,
    tied_tol: float = 1e-8,
) -> Dict[str, object]:
    """Evaluate ``w * pathology + (1-w) * molecular`` for every weight."""

    if weights is None:
        weights = np.linspace(0.0, 1.0, 21)
    weights = np.asarray(list(weights), dtype=np.float64)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("weights must be a non-empty one-dimensional sequence")
    if not np.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
        raise ValueError("weights must be finite values in [0, 1]")

    pairs = build_fold_pairs(frame, tied_tol=tied_tol)
    path = vectors_from_column(pairs, "_pathology_z")
    molecular = vectors_from_column(pairs, "_molecular_z")
    rows = []
    half_vectors = None
    for weight in weights:
        vectors = {
            fold: float(weight) * path[fold] + (1.0 - float(weight)) * molecular[fold]
            for fold in pairs
        }
        result = evaluate_vectors(pairs, vectors, tied_tol)
        rows.append(
            {
                "w": float(weight),
                "c_index_pooled": result["c_index_pooled"],
                "c_index_foldmean": result["c_index_foldmean"],
                "c_index_foldstd_ddof0": result["c_index_foldstd_ddof0"],
                "R_P": result["group_accuracy"][GROUP_P],
                "R_M": result["group_accuracy"][GROUP_M],
                "n_pairs": result["n_pairs"],
            }
        )
        if np.isclose(weight, 0.5, rtol=0.0, atol=1e-12):
            half_vectors = vectors

    if half_vectors is None:
        raise ValueError("weights must include w=0.5 for the Simple Fusion check")
    simple_vectors = vectors_from_column(pairs, "simple_fusion_risk")
    max_risk_difference = max(
        float(np.max(np.abs(half_vectors[fold] - simple_vectors[fold]))) for fold in pairs
    )
    half_result = evaluate_vectors(pairs, half_vectors, tied_tol)
    simple_result = evaluate_vectors(pairs, simple_vectors, tied_tol)
    fused_result = evaluate_vectors(pairs, vectors_from_column(pairs, "fused_risk"), tied_tol)
    curve = pd.DataFrame(rows).sort_values("w").reset_index(drop=True)
    best_index = int(curve["c_index_pooled"].idxmax())
    best = curve.loc[best_index]
    summary = pd.DataFrame(
        [
            {
                "best_w": float(best["w"]),
                "best_global_weight_c_index_pooled": float(best["c_index_pooled"]),
                "simple_fusion_c_index_pooled": simple_result["c_index_pooled"],
                "fused_c_index_pooled": fused_result["c_index_pooled"],
                "fused_exceeds_best_global_weight": bool(
                    fused_result["c_index_pooled"] > best["c_index_pooled"]
                ),
            }
        ]
    )
    checks = {
        "w_0_5_max_abs_risk_difference": max_risk_difference,
        "w_0_5_c_index_difference": abs(
            half_result["c_index_pooled"] - simple_result["c_index_pooled"]
        ),
        "status": "PASS"
        if max_risk_difference <= 1e-12
        and abs(half_result["c_index_pooled"] - simple_result["c_index_pooled"]) <= 1e-12
        else "FAIL",
    }
    return {"curve": curve, "summary": summary, "checks": checks}
