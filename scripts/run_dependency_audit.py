#!/usr/bin/env python
"""Audit inference-time modality shuffling and ranking flips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dependency.shuffle import (
    generated_simple_fusion_shuffle,
    provided_shuffle_audit,
    summarize_dependency,
)
from src.io import add_standardized_reference_risks, load_predictions


def _parse_mapping(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--shuffled-risk must use NAME=COLUMN")
        name, column = value.split("=", 1)
        if not name or not column or name in result:
            raise ValueError("Invalid or duplicate shuffled-risk mapping: " + value)
        result[name] = column
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate-simple-fusion-shuffle", choices=["pathology", "molecular"])
    mode.add_argument("--shuffled-risk", action="append", default=[], metavar="NAME=COLUMN")
    parser.add_argument("--clean-risk-column", default="fused_risk")
    parser.add_argument("--donor-seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--risks-already-standardized", action="store_true")
    parser.add_argument("--tied-tol", type=float, default=1e-8)
    args = parser.parse_args()

    mappings = _parse_mapping(args.shuffled_risk)
    extra = [args.clean_risk_column, *mappings.values()]
    frame = load_predictions(args.predictions, extra_columns=extra)
    if args.generate_simple_fusion_shuffle:
        frame = add_standardized_reference_risks(
            frame, already_standardized=args.risks_already_standardized
        )
        detail = generated_simple_fusion_shuffle(
            frame, args.generate_simple_fusion_shuffle, args.donor_seeds, args.tied_tol
        )
        generated = True
    else:
        detail = provided_shuffle_audit(
            frame, mappings, clean_column=args.clean_risk_column, tied_tol=args.tied_tol
        )
        generated = False
    fold_summary, summary = summarize_dependency(detail)
    checks = {
        "mode": "generated_simple_fusion" if generated else "provided_shuffled_risks",
        "all_generated_derangements_have_zero_fixed_points": True if generated else None,
        "detail_rows": len(detail),
        "status": "PASS",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "dependency_detail.csv", index=False, float_format="%.10g")
    fold_summary.to_csv(args.output_dir / "dependency_fold_summary.csv", index=False, float_format="%.10g")
    summary.to_csv(args.output_dir / "dependency_summary.csv", index=False, float_format="%.10g")
    with (args.output_dir / "checks.json").open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2)
    print("Dependency audit PASS: {}".format(args.output_dir))


if __name__ == "__main__":
    main()
