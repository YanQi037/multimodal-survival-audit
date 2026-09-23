"""Core fold-local Harrell pair metrics."""

from .pairs import (
    DEFAULT_TIED_TOL,
    GROUP_EMPTY,
    GROUP_M,
    GROUP_P,
    GROUP_PM,
    GROUP_TIE,
    build_fold_pairs,
    evaluate_vectors,
    five_group_decomposition,
    ranking_credit,
)

__all__ = [
    "DEFAULT_TIED_TOL",
    "GROUP_EMPTY",
    "GROUP_M",
    "GROUP_P",
    "GROUP_PM",
    "GROUP_TIE",
    "build_fold_pairs",
    "evaluate_vectors",
    "five_group_decomposition",
    "ranking_credit",
]

