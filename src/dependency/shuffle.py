"""Formal deterministic derangement and dependency metrics.

The mapping and ranking-flip denominator follow the final MCAT/MOTCat
``eval_modality_dependency.py`` implementation. Ranking flips use all patient
pairs that are strictly ordered by both clean and shuffled predictions; they do
not use survival outcomes.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sksurv.metrics import concordance_index_censored


def deterministic_patient_derangement(
    case_ids: Iterable[str], fold: int, shuffle_seed: int
) -> tuple[np.ndarray, Dict[str, str]]:
    """Return the formal single-cycle ``motcat_patient_shuffle_v1`` mapping."""

    case_ids = tuple(str(case_id) for case_id in case_ids)
    if len(case_ids) < 2:
        raise ValueError("Patient-level shuffle requires at least two patients")
    payload = (
        "motcat_patient_shuffle_v1|fold={}|seed={}|cases={}|ids={}".format(
            int(fold), int(shuffle_seed), len(case_ids), "\x1f".join(case_ids)
        )
    ).encode("utf-8")
    seed64 = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    rng = np.random.Generator(np.random.PCG64(seed64))
    cycle_order = rng.permutation(len(case_ids))
    donor_indices = np.empty(len(case_ids), dtype=np.int64)
    donor_indices[cycle_order] = np.roll(cycle_order, -1)
    if np.any(donor_indices == np.arange(len(case_ids))):
        raise AssertionError("Derangement contains fixed points")
    donor_map = {
        case_ids[recipient]: case_ids[int(donor_indices[recipient])]
        for recipient in range(len(case_ids))
    }
    if len(set(donor_map.values())) != len(case_ids):
        raise AssertionError("Donor mapping is not one-to-one")
    return donor_indices, donor_map


def dependency_metrics(clean: np.ndarray, shuffled: np.ndarray) -> dict:
    """Compute the formal effect, correlation, and ranking-flip metrics."""

    clean = np.asarray(clean, dtype=np.float64)
    shuffled = np.asarray(shuffled, dtype=np.float64)
    if clean.ndim != 1 or clean.shape != shuffled.shape:
        raise ValueError("Risk arrays must be aligned one-dimensional vectors")
    if not np.isfinite(clean).all() or not np.isfinite(shuffled).all():
        raise ValueError("Risk arrays contain NaN/Inf")
    delta = np.abs(shuffled - clean)
    row_i, row_j = np.triu_indices(len(clean), k=1)
    clean_diff = clean[row_i] - clean[row_j]
    shuffled_diff = shuffled[row_i] - shuffled[row_j]
    orderable = (np.abs(clean_diff) > 1e-12) & (
        np.abs(shuffled_diff) > 1e-12
    )
    flipped = orderable & ((clean_diff * shuffled_diff) < 0)
    n_orderable = int(orderable.sum())
    clean_std = float(np.std(clean, ddof=0))
    mean_abs = float(np.mean(delta))
    return {
        "pearson_r": float(pearsonr(clean, shuffled).statistic),
        "spearman_r": float(spearmanr(clean, shuffled).statistic),
        "mean_abs_risk_difference": mean_abs,
        "median_abs_risk_difference": float(np.median(delta)),
        "clean_risk_std": clean_std,
        "standardized_mean_abs_effect": (
            float(mean_abs / clean_std) if clean_std > 0 else np.nan
        ),
        "ranking_flip_rate": (
            float(flipped.sum() / n_orderable) if n_orderable else np.nan
        ),
        "n_orderable_pairs": n_orderable,
        "n_flipped_pairs": int(flipped.sum()),
    }


def _c_index(frame: pd.DataFrame, risk: np.ndarray, tied_tol: float) -> float:
    return float(
        concordance_index_censored(
            frame["event"].to_numpy(dtype=int).astype(bool),
            frame["time"].to_numpy(dtype=float),
            np.asarray(risk, dtype=float),
            tied_tol=tied_tol,
        )[0]
    )


def _detail_row(
    fold_frame: pd.DataFrame,
    clean: np.ndarray,
    shuffled: np.ndarray,
    *,
    fold: int,
    condition: str,
    donor_seed: int | None,
    tied_tol: float,
) -> dict:
    metrics = dependency_metrics(clean, shuffled)
    clean_c = _c_index(fold_frame, clean, tied_tol)
    shuffled_c = _c_index(fold_frame, shuffled, tied_tol)
    return {
        "condition": condition,
        "fold": fold,
        "donor_seed": donor_seed,
        "n_patients": len(fold_frame),
        "clean_c_index": clean_c,
        "shuffled_c_index": shuffled_c,
        "delta_c_index": shuffled_c - clean_c,
        **metrics,
    }


def generated_simple_fusion_shuffle(
    frame: pd.DataFrame,
    modality: str,
    donor_seeds: Iterable[int],
    tied_tol: float = 1e-8,
) -> pd.DataFrame:
    """Shuffle one standardized reference and re-evaluate Simple Fusion."""

    if modality not in {"pathology", "molecular"}:
        raise ValueError("modality must be pathology or molecular")
    rows = []
    for fold in sorted(frame["fold"].unique()):
        fold_frame = frame[frame["fold"] == fold].sort_values("patient_id").reset_index(drop=True)
        pathology = fold_frame["_pathology_z"].to_numpy(float)
        molecular = fold_frame["_molecular_z"].to_numpy(float)
        clean = 0.5 * (pathology + molecular)
        ids = fold_frame["patient_id"].astype(str).tolist()
        for donor_seed in donor_seeds:
            donor_indices, _ = deterministic_patient_derangement(ids, int(fold), int(donor_seed))
            if modality == "pathology":
                shuffled = 0.5 * (pathology[donor_indices] + molecular)
            else:
                shuffled = 0.5 * (pathology + molecular[donor_indices])
            rows.append(
                _detail_row(
                    fold_frame,
                    clean,
                    shuffled,
                    fold=int(fold),
                    condition="{}_shuffle_simple_fusion".format(modality),
                    donor_seed=int(donor_seed),
                    tied_tol=tied_tol,
                )
            )
    return pd.DataFrame(rows)


def provided_shuffle_audit(
    frame: pd.DataFrame,
    shuffled_columns: Mapping[str, str],
    clean_column: str = "fused_risk",
    tied_tol: float = 1e-8,
) -> pd.DataFrame:
    """Audit precomputed inference-time shuffled risks stored in the CSV."""

    rows = []
    for fold in sorted(frame["fold"].unique()):
        fold_frame = frame[frame["fold"] == fold].sort_values("patient_id").reset_index(drop=True)
        clean = fold_frame[clean_column].to_numpy(float)
        for name, column in shuffled_columns.items():
            rows.append(
                _detail_row(
                    fold_frame,
                    clean,
                    fold_frame[column].to_numpy(float),
                    fold=int(fold),
                    condition=name,
                    donor_seed=None,
                    tied_tol=tied_tol,
                )
            )
    return pd.DataFrame(rows)


def summarize_dependency(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average seeds within fold, then summarize the fold-level values."""

    metrics = [
        "clean_c_index",
        "shuffled_c_index",
        "delta_c_index",
        "pearson_r",
        "spearman_r",
        "mean_abs_risk_difference",
        "median_abs_risk_difference",
        "standardized_mean_abs_effect",
        "ranking_flip_rate",
    ]
    fold_summary = (
        detail.groupby(["condition", "fold"], as_index=False)[metrics]
        .mean()
        .sort_values(["condition", "fold"])
    )
    rows = []
    for condition, group in fold_summary.groupby("condition", sort=True):
        row = {"condition": condition, "n_folds": len(group)}
        for metric in metrics:
            row["mean_{}".format(metric)] = float(group[metric].mean())
            row["std_{}_ddof0".format(metric)] = float(group[metric].std(ddof=0))
        rows.append(row)
    return fold_summary, pd.DataFrame(rows)


