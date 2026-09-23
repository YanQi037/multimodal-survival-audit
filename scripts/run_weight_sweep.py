#!/usr/bin/env python
"""Run the pooled global pathology-weight sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.global_weight.sweep import run_weight_sweep
from src.io import add_standardized_reference_risks, load_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--risks-already-standardized", action="store_true")
    parser.add_argument("--tied-tol", type=float, default=1e-8)
    args = parser.parse_args()
    if args.step <= 0 or args.step > 1 or abs(round(1.0 / args.step) * args.step - 1.0) > 1e-12:
        raise ValueError("--step must be positive and divide 1.0 exactly")
    count = int(round(1.0 / args.step))
    weights = [index / count for index in range(count + 1)]

    frame = add_standardized_reference_risks(
        load_predictions(args.predictions, extra_columns=["fused_risk"]),
        already_standardized=args.risks_already_standardized,
    )
    result = run_weight_sweep(frame, weights=weights, tied_tol=args.tied_tol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result["curve"].to_csv(args.output_dir / "weight_sweep_curve.csv", index=False, float_format="%.10g")
    result["summary"].to_csv(args.output_dir / "weight_sweep_summary.csv", index=False, float_format="%.10g")
    with (args.output_dir / "checks.json").open("w", encoding="utf-8") as handle:
        json.dump(result["checks"], handle, indent=2)
    print("Weight sweep {}: {}".format(result["checks"]["status"], args.output_dir))


if __name__ == "__main__":
    main()
