"""Formal fold-local pair construction and pooled C-index decomposition.

Adapted with minimal changes from the formal Survival project implementations:
``results/research_restart/13_weight_sweep/build_weight_sweep.py`` and stages
16/17. Higher risk always means worse prognosis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pandas as pd
from sksurv.metrics import _iter_comparable, concordance_index_censored


DEFAULT_TIED_TOL = 1e-8
GROUP_PM = "PM"
GROUP_P = "P"
GROUP_M = "M"
GROUP_EMPTY = "empty"
GROUP_TIE = "tie"
GROUPS = (GROUP_PM, GROUP_P, GROUP_M, GROUP_EMPTY, GROUP_TIE)


@dataclass
class FoldPairs:
    fold: int
    frame: pd.DataFrame
    event_indices: np.ndarray
    other_indices: np.ndarray
    groups: np.ndarray

    @property
    def n_pairs(self) -> int:
        return int(len(self.groups))


def ranking_credit(
    risk: np.ndarray,
    event_indices: np.ndarray,
    other_indices: np.ndarray,
    tied_tol: float = DEFAULT_TIED_TOL,
) -> np.ndarray:
    """Return pair credit: correct=1, risk tie=0.5, wrong=0."""

    risk = np.asarray(risk, dtype=np.float64)
    difference = risk[event_indices] - risk[other_indices]
    result = np.zeros(len(difference), dtype=np.float64)
    ties = np.abs(difference) <= tied_tol
    result[ties] = 0.5
    result[(difference > 0.0) & ~ties] = 1.0
    return result


def _comparable_indices(event: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(time)
    event_indices = []
    other_indices = []
    for index_in_order, mask, _ in _iter_comparable(event, time, order):
        event_index = int(order[index_in_order])
        others = order[mask]
        event_indices.extend([event_index] * len(others))
        other_indices.extend(int(value) for value in others)
    if not event_indices:
        raise ValueError("A fold contains no Harrell-comparable pairs")
    return np.asarray(event_indices, dtype=np.int64), np.asarray(other_indices, dtype=np.int64)


def build_fold_pairs(
    frame: pd.DataFrame,
    *,
    pathology_column: str = "_pathology_z",
    molecular_column: str = "_molecular_z",
    tied_tol: float = DEFAULT_TIED_TOL,
) -> Dict[int, FoldPairs]:
    """Build the five mutually exclusive reference-defined pair groups."""

    result: Dict[int, FoldPairs] = {}
    for fold in sorted(frame["fold"].unique()):
        fold_frame = frame[frame["fold"] == fold].sort_values("patient_id").reset_index(drop=True)
        event = fold_frame["event"].to_numpy(dtype=np.int64).astype(bool)
        time = fold_frame["time"].to_numpy(dtype=np.float64)
        event_indices, other_indices = _comparable_indices(event, time)
        path_score = ranking_credit(
            fold_frame[pathology_column].to_numpy(float), event_indices, other_indices, tied_tol
        )
        molecular_score = ranking_credit(
            fold_frame[molecular_column].to_numpy(float), event_indices, other_indices, tied_tol
        )
        groups = np.full(len(event_indices), GROUP_TIE, dtype=object)
        strict = (path_score != 0.5) & (molecular_score != 0.5)
        groups[strict & (path_score == 1.0) & (molecular_score == 1.0)] = GROUP_PM
        groups[strict & (path_score == 1.0) & (molecular_score == 0.0)] = GROUP_P
        groups[strict & (path_score == 0.0) & (molecular_score == 1.0)] = GROUP_M
        groups[strict & (path_score == 0.0) & (molecular_score == 0.0)] = GROUP_EMPTY
        result[int(fold)] = FoldPairs(
            fold=int(fold),
            frame=fold_frame,
            event_indices=event_indices,
            other_indices=other_indices,
            groups=groups,
        )
    return result


def vectors_from_column(pairs: Mapping[int, FoldPairs], column: str) -> Dict[int, np.ndarray]:
    return {fold: data.frame[column].to_numpy(dtype=np.float64) for fold, data in pairs.items()}


def evaluate_vectors(
    pairs: Mapping[int, FoldPairs],
    vectors: Mapping[int, np.ndarray],
    tied_tol: float = DEFAULT_TIED_TOL,
) -> Dict[str, object]:
    """Compute fold C-indices, pooled C-index, and pooled group accuracies."""

    total_credit = 0.0
    total_pairs = 0
    group_counts = {group: 0 for group in GROUPS}
    group_credits = {group: 0.0 for group in GROUPS}
    fold_rows = []
    max_direct_error = 0.0
    for fold, data in pairs.items():
        risk = np.asarray(vectors[fold], dtype=np.float64)
        if len(risk) != len(data.frame) or not np.isfinite(risk).all():
            raise ValueError("Invalid risk vector for fold {}".format(fold))
        score = ranking_credit(risk, data.event_indices, data.other_indices, tied_tol)
        c_pair = float(score.mean())
        c_direct = float(
            concordance_index_censored(
                data.frame["event"].to_numpy(dtype=int).astype(bool),
                data.frame["time"].to_numpy(dtype=float),
                risk,
                tied_tol=tied_tol,
            )[0]
        )
        max_direct_error = max(max_direct_error, abs(c_pair - c_direct))
        fold_rows.append({"fold": fold, "n_pairs": len(score), "c_index": c_pair})
        total_credit += float(score.sum())
        total_pairs += int(len(score))
        for group in GROUPS:
            mask = data.groups == group
            group_counts[group] += int(mask.sum())
            group_credits[group] += float(score[mask].sum())
    if total_pairs == 0:
        raise ValueError("No comparable pairs")
    group_accuracy = {
        group: (float(group_credits[group] / group_counts[group]) if group_counts[group] else np.nan)
        for group in GROUPS
    }
    group_proportion = {group: float(group_counts[group] / total_pairs) for group in GROUPS}
    group_contribution = {group: float(group_credits[group] / total_pairs) for group in GROUPS}
    pooled = float(total_credit / total_pairs)
    decomposition_error = abs(pooled - sum(group_contribution.values()))
    return {
        "c_index_pooled": pooled,
        "c_index_foldmean": float(np.mean([row["c_index"] for row in fold_rows])),
        "c_index_foldstd_ddof0": float(np.std([row["c_index"] for row in fold_rows], ddof=0)),
        "n_pairs": total_pairs,
        "group_counts": group_counts,
        "group_proportion": group_proportion,
        "group_accuracy": group_accuracy,
        "group_contribution": group_contribution,
        "fold_rows": fold_rows,
        "decomposition_error": decomposition_error,
        "direct_cindex_error": max_direct_error,
    }


def five_group_decomposition(fused: Dict[str, object], simple: Dict[str, object]) -> Dict[str, object]:
    """Decompose pooled ``C_fused-C_simple`` into five pair-group terms."""

    if fused["group_counts"] != simple["group_counts"] or fused["n_pairs"] != simple["n_pairs"]:
        raise ValueError("Fused and Simple Fusion must share the reference-defined pair universe")
    terms = {
        group: float(fused["group_contribution"][group] - simple["group_contribution"][group])
        for group in GROUPS
    }
    delta = float(fused["c_index_pooled"] - simple["c_index_pooled"])
    complementary = terms[GROUP_P] + terms[GROUP_M]
    residual = terms[GROUP_PM] + terms[GROUP_EMPTY] + terms[GROUP_TIE]
    return {
        "delta_c_pooled": delta,
        "terms": terms,
        "complementary_term": float(complementary),
        "residual": float(residual),
        "identity_error": abs(delta - sum(terms.values())),
    }


