#!/usr/bin/env python
"""Run the less-used-modality Simple Fusion permutation-null audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.io import add_standardized_reference_risks, load_predictions
from src.permutation.null import permutation_null_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dominant", required=True, choices=["pathology", "molecular"])
    parser.add_argument("--donor-seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--setting-name", default=None)
    parser.add_argument("--risks-already-standardized", action="store_true")
    parser.add_argument("--tied-tol", type=float, default=1e-8)
    args = parser.parse_args()

    frame = add_standardized_reference_risks(
        load_predictions(args.predictions),
        already_standardized=args.risks_already_standardized,
    )
    result = permutation_null_audit(
        frame,
        dominant=args.dominant,
        donor_seeds=args.donor_seeds,
        setting_name=args.setting_name or args.predictions.stem,
        tied_tol=args.tied_tol,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result["detail"].to_csv(args.output_dir / "permutation_fold_seed.csv", index=False, float_format="%.10g")
    result["fold_summary"].to_csv(args.output_dir / "permutation_fold_summary.csv", index=False, float_format="%.10g")
    result["summary"].to_csv(args.output_dir / "permutation_summary.csv", index=False, float_format="%.10g")
    with (args.output_dir / "checks.json").open("w", encoding="utf-8") as handle:
        json.dump(result["checks"], handle, indent=2)
    print("Permutation-null audit {}: {}".format(result["checks"]["status"], args.output_dir))


if __name__ == "__main__":
    main()
