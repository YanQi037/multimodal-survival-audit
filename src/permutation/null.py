"""Formal less-used-modality permutation null from research-restart stage 15."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd
from sksurv.metrics import _iter_comparable, concordance_index_censored

from src.metrics.pairs import ranking_credit


def _stable_setting_code(setting_name: str) -> tuple[int, int]:
    digest = hashlib.sha256(setting_name.encode("utf-8")).digest()
    return (
        int.from_bytes(digest[:4], byteorder="little", signed=False),
        int.from_bytes(digest[4:8], byteorder="little", signed=False),
    )


def sattolo_derangement(
    n_items: int, donor_seed: int, fold: int, setting_name: str
) -> np.ndarray:
    """Return the deterministic stage-15 single-cycle derangement."""

    if n_items < 2:
        raise ValueError("A derangement requires at least two patients")
    code_a, code_b = _stable_setting_code(setting_name)
    rng = np.random.default_rng(
        np.random.SeedSequence([donor_seed, fold, code_a, code_b])
    )
    permutation = np.arange(n_items, dtype=np.int64)
    for index in range(n_items - 1, 0, -1):
        swap_index = int(rng.integers(0, index))
        permutation[index], permutation[swap_index] = (
            permutation[swap_index],
            permutation[index],
        )
    if np.any(permutation == np.arange(n_items)):
        raise AssertionError("Derangement contains fixed points")
    if len(np.unique(permutation)) != n_items:
        raise AssertionError("Donor mapping is not a permutation")
    return permutation


def _comparable(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    event = frame["event"].to_numpy(dtype=int).astype(bool)
    time = frame["time"].to_numpy(dtype=float)
    order = np.argsort(time)
    event_indices = []
    other_indices = []
    for index_in_order, mask, _ in _iter_comparable(event, time, order):
        event_index = int(order[index_in_order])
        others = order[mask]
        event_indices.extend([event_index] * len(others))
        other_indices.extend(int(value) for value in others)
    if not event_indices:
        raise ValueError("A fold contains no comparable pairs")
    return np.asarray(event_indices, int), np.asarray(other_indices, int)


def _cindex(frame: pd.DataFrame, risk: np.ndarray, tied_tol: float) -> float:
    return float(
        concordance_index_censored(
            frame["event"].to_numpy(dtype=int).astype(bool),
            frame["time"].to_numpy(float),
            np.asarray(risk, float),
            tied_tol=tied_tol,
        )[0]
    )


def _pair_diagnostics(
    pathology: np.ndarray,
    molecular: np.ndarray,
    event_indices: np.ndarray,
    other_indices: np.ndarray,
    tied_tol: float,
) -> dict:
    p = ranking_credit(pathology, event_indices, other_indices, tied_tol)
    m = ranking_credit(molecular, event_indices, other_indices, tied_tol)
    tie = (p == 0.5) | (m == 0.5)
    strict = ~tie
    p_only = strict & (p == 1.0) & (m == 0.0)
    m_only = strict & (p == 0.0) & (m == 1.0)
    return {
        "n_pairs": len(p),
        "n_P": int(p_only.sum()),
        "n_M": int(m_only.sum()),
        "n_tie": int(tie.sum()),
        "pi_P": float(p_only.mean()),
        "pi_M": float(m_only.mean()),
        "oracle_c_index": float(np.maximum(p, m).mean()),
    }


def permutation_null_audit(
    frame: pd.DataFrame,
    *,
    dominant: str,
    donor_seeds: Iterable[int] = (11, 22, 33, 44, 55),
    setting_name: str = "prediction_audit",
    tied_tol: float = 1e-8,
) -> dict:
    """Compare real Simple Fusion with a shuffled less-used reference."""

    if dominant not in {"pathology", "molecular"}:
        raise ValueError("dominant must be pathology or molecular")
    donor_seeds = tuple(int(seed) for seed in donor_seeds)
    if not donor_seeds:
        raise ValueError("At least one donor seed is required")
    rows = []
    for fold in sorted(frame["fold"].unique()):
        fold_frame = frame[frame["fold"] == fold].sort_values("patient_id").reset_index(drop=True)
        pathology = fold_frame["_pathology_z"].to_numpy(float)
        molecular = fold_frame["_molecular_z"].to_numpy(float)
        real_sf = 0.5 * (pathology + molecular)
        if dominant == "pathology":
            dominant_risk, less_risk = pathology, molecular
        else:
            dominant_risk, less_risk = molecular, pathology
        event_indices, other_indices = _comparable(fold_frame)
        real_pair = _pair_diagnostics(
            pathology, molecular, event_indices, other_indices, tied_tol
        )
        c_dominant = _cindex(fold_frame, dominant_risk, tied_tol)
        c_real = _cindex(fold_frame, real_sf, tied_tol)
        for seed in donor_seeds:
            permutation = sattolo_derangement(
                len(fold_frame), seed, int(fold), setting_name
            )
            shuffled_less = less_risk[permutation]
            null_sf = 0.5 * (dominant_risk + shuffled_less)
            if dominant == "pathology":
                null_pathology, null_molecular = dominant_risk, shuffled_less
            else:
                null_pathology, null_molecular = shuffled_less, dominant_risk
            null_pair = _pair_diagnostics(
                null_pathology,
                null_molecular,
                event_indices,
                other_indices,
                tied_tol,
            )
            c_null = _cindex(fold_frame, null_sf, tied_tol)
            rows.append(
                {
                    "fold": int(fold),
                    "donor_seed": seed,
                    "n_patients": len(fold_frame),
                    "dominant": dominant,
                    "less_used": (
                        "molecular" if dominant == "pathology" else "pathology"
                    ),
                    "c_dominant": c_dominant,
                    "c_sf_real": c_real,
                    "delta_real": c_real - c_dominant,
                    "c_sf_null": c_null,
                    "delta_null": c_null - c_dominant,
                    "gap_real_vs_null": c_real - c_null,
                    "real_pi_P": real_pair["pi_P"],
                    "real_pi_M": real_pair["pi_M"],
                    "real_oracle": real_pair["oracle_c_index"],
                    "null_pi_P": null_pair["pi_P"],
                    "null_pi_M": null_pair["pi_M"],
                    "null_oracle": null_pair["oracle_c_index"],
                    "n_pairs": real_pair["n_pairs"],
                    "fixed_points": 0,
                }
            )
    detail = pd.DataFrame(rows).sort_values(["fold", "donor_seed"])
    average_columns = [
        "c_sf_null",
        "delta_null",
        "gap_real_vs_null",
        "null_pi_P",
        "null_pi_M",
        "null_oracle",
    ]
    constant_columns = [
        "c_dominant",
        "c_sf_real",
        "delta_real",
        "real_pi_P",
        "real_pi_M",
        "real_oracle",
        "n_pairs",
    ]
    fold_rows = []
    for fold, group in detail.groupby("fold", sort=True):
        row = {"fold": int(fold)}
        for column in constant_columns:
            if group[column].nunique() != 1:
                raise AssertionError("{} changed across seeds".format(column))
            row[column] = group.iloc[0][column]
        for column in average_columns:
            row["mean_{}_over_seeds".format(column)] = float(group[column].mean())
        row["real_sf_gt_dominant"] = int(row["delta_real"] > 1e-12)
        row["null_sf_gt_dominant"] = int(
            row["mean_delta_null_over_seeds"] > 1e-12
        )
        fold_rows.append(row)
    fold_summary = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(
        [
            {
                "n_folds": len(fold_summary),
                "real_sf_gt_dominant_folds": int(
                    fold_summary["real_sf_gt_dominant"].sum()
                ),
                "null_sf_gt_dominant_folds": int(
                    fold_summary["null_sf_gt_dominant"].sum()
                ),
                "mean_delta_real": float(fold_summary["delta_real"].mean()),
                "mean_delta_null": float(
                    fold_summary["mean_delta_null_over_seeds"].mean()
                ),
                "mean_gap_real_vs_null": float(
                    fold_summary["mean_gap_real_vs_null_over_seeds"].mean()
                ),
                "mean_c_sf_real": float(fold_summary["c_sf_real"].mean()),
                "mean_c_sf_null": float(
                    fold_summary["mean_c_sf_null_over_seeds"].mean()
                ),
                "real_pi_P_foldmean": float(fold_summary["real_pi_P"].mean()),
                "null_pi_P_foldmean": float(
                    fold_summary["mean_null_pi_P_over_seeds"].mean()
                ),
                "real_pi_M_foldmean": float(fold_summary["real_pi_M"].mean()),
                "null_pi_M_foldmean": float(
                    fold_summary["mean_null_pi_M_over_seeds"].mean()
                ),
                "real_oracle_foldmean": float(fold_summary["real_oracle"].mean()),
                "null_oracle_foldmean": float(
                    fold_summary["mean_null_oracle_over_seeds"].mean()
                ),
            }
        ]
    )
    checks = {
        "all_derangements_have_zero_fixed_points": bool(
            (detail["fixed_points"] == 0).all()
        ),
        "detail_rows": len(detail),
        "fold_rows": len(fold_summary),
        "donor_seeds": list(donor_seeds),
        "status": "PASS",
    }
    return {
        "detail": detail,
        "fold_summary": fold_summary,
        "summary": summary,
        "checks": checks,
    }

