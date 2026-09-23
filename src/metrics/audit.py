"""High-level pair audit assembled from the formal pair primitives."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .pairs import (
    GROUPS,
    GROUP_EMPTY,
    GROUP_M,
    GROUP_P,
    GROUP_PM,
    GROUP_TIE,
    build_fold_pairs,
    evaluate_vectors,
    five_group_decomposition,
    ranking_credit,
    vectors_from_column,
)


def run_pair_audit(frame: pd.DataFrame, tied_tol: float = 1e-8) -> Dict[str, object]:
    """Compute pooled pair partition, rescue, decomposition, and fold stability."""

    pairs = build_fold_pairs(frame, tied_tol=tied_tol)
    simple_vectors = vectors_from_column(pairs, "simple_fusion_risk")
    fused_vectors = vectors_from_column(pairs, "fused_risk")
    simple = evaluate_vectors(pairs, simple_vectors, tied_tol)
    fused = evaluate_vectors(pairs, fused_vectors, tied_tol)
    decomposition = five_group_decomposition(fused, simple)

    partition_rows = []
    for group in GROUPS:
        partition_rows.append(
            {
                "group": group,
                "n_pairs": fused["group_counts"][group],
                "proportion": fused["group_proportion"][group],
                "A_fused": fused["group_accuracy"][group],
                "A_simple_fusion": simple["group_accuracy"][group],
                "term": decomposition["terms"][group],
            }
        )

    fold_rows = []
    for fold, data in pairs.items():
        score = ranking_credit(
            fused_vectors[fold], data.event_indices, data.other_indices, tied_tol
        )
        p_mask = data.groups == GROUP_P
        m_mask = data.groups == GROUP_M
        if not p_mask.any() or not m_mask.any():
            raise ValueError(
                "Fold {} has an empty P or M rescue opportunity group".format(fold)
            )
        r_p = float(score[p_mask].mean())
        r_m = float(score[m_mask].mean())
        fold_rows.append(
            {
                "fold": fold,
                "n_pairs": data.n_pairs,
                "n_P": int(p_mask.sum()),
                "n_M": int(m_mask.sum()),
                "R_P": r_p,
                "R_M": r_m,
                "R_P_minus_R_M": r_p - r_m,
            }
        )
    fold_frame = pd.DataFrame(fold_rows).sort_values("fold")
    pooled_difference = (
        fused["group_accuracy"][GROUP_P] - fused["group_accuracy"][GROUP_M]
    )
    pooled_sign = int(np.sign(pooled_difference))
    fold_sign = np.sign(fold_frame["R_P_minus_R_M"].to_numpy(float)).astype(int)
    rescue_summary = {
        "pooled_R_P": fused["group_accuracy"][GROUP_P],
        "pooled_R_M": fused["group_accuracy"][GROUP_M],
        "pooled_R_P_minus_R_M": pooled_difference,
        "pooled_sign": pooled_sign,
        "folds_same_sign": int(np.count_nonzero(fold_sign == pooled_sign)),
        "folds_opposite_sign": (
            int(np.count_nonzero(fold_sign == -pooled_sign)) if pooled_sign else 0
        ),
        "R_P_min": float(fold_frame["R_P"].min()),
        "R_P_max": float(fold_frame["R_P"].max()),
        "R_M_min": float(fold_frame["R_M"].min()),
        "R_M_max": float(fold_frame["R_M"].max()),
    }

    pm_simple = simple["group_accuracy"][GROUP_PM]
    empty_simple = simple["group_accuracy"][GROUP_EMPTY]
    checks = {
        "five_group_proportion_sum_error": abs(
            sum(fused["group_proportion"].values()) - 1.0
        ),
        "fused_cindex_decomposition_error": fused["decomposition_error"],
        "simple_cindex_decomposition_error": simple["decomposition_error"],
        "delta_five_term_identity_error": decomposition["identity_error"],
        "simple_A_PM_equals_one": bool(pm_simple == 1.0),
        "simple_A_empty_equals_zero": bool(empty_simple == 0.0),
        "maximum_direct_cindex_error": max(
            fused["direct_cindex_error"], simple["direct_cindex_error"]
        ),
    }
    checks["status"] = "PASS" if (
        max(
            checks["five_group_proportion_sum_error"],
            checks["fused_cindex_decomposition_error"],
            checks["simple_cindex_decomposition_error"],
            checks["delta_five_term_identity_error"],
            checks["maximum_direct_cindex_error"],
        ) <= 1e-12
        and checks["simple_A_PM_equals_one"]
        and checks["simple_A_empty_equals_zero"]
    ) else "FAIL"

    metrics = pd.DataFrame(
        [
            {
                "method": "fused",
                "c_index_pooled": fused["c_index_pooled"],
                "c_index_foldmean": fused["c_index_foldmean"],
                "c_index_foldstd_ddof0": fused["c_index_foldstd_ddof0"],
                "R_P": fused["group_accuracy"][GROUP_P],
                "R_M": fused["group_accuracy"][GROUP_M],
                "A_PM": fused["group_accuracy"][GROUP_PM],
                "A_empty": fused["group_accuracy"][GROUP_EMPTY],
                "A_tie": fused["group_accuracy"][GROUP_TIE],
            },
            {
                "method": "simple_fusion",
                "c_index_pooled": simple["c_index_pooled"],
                "c_index_foldmean": simple["c_index_foldmean"],
                "c_index_foldstd_ddof0": simple["c_index_foldstd_ddof0"],
                "R_P": simple["group_accuracy"][GROUP_P],
                "R_M": simple["group_accuracy"][GROUP_M],
                "A_PM": simple["group_accuracy"][GROUP_PM],
                "A_empty": simple["group_accuracy"][GROUP_EMPTY],
                "A_tie": simple["group_accuracy"][GROUP_TIE],
            },
        ]
    )
    decomposition_row = {
        "C_fused_pooled": fused["c_index_pooled"],
        "C_simple_fusion_pooled": simple["c_index_pooled"],
        "delta_C_pooled": decomposition["delta_c_pooled"],
        "term_PM": decomposition["terms"][GROUP_PM],
        "term_P": decomposition["terms"][GROUP_P],
        "term_M": decomposition["terms"][GROUP_M],
        "term_empty": decomposition["terms"][GROUP_EMPTY],
        "term_tie": decomposition["terms"][GROUP_TIE],
        "complementary_term": decomposition["complementary_term"],
        "residual": decomposition["residual"],
        "identity_error": decomposition["identity_error"],
    }
    return {
        "partition": pd.DataFrame(partition_rows),
        "metrics": metrics,
        "decomposition": pd.DataFrame([decomposition_row]),
        "fold_rescue": fold_frame,
        "rescue_summary": pd.DataFrame([rescue_summary]),
        "checks": checks,
    }

