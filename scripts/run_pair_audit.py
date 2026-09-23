#!/usr/bin/env python
"""Run complementary-pair, decomposition, and rescue-stability audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.io import add_standardized_reference_risks, load_predictions
from src.metrics.audit import run_pair_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--risks-already-standardized", action="store_true")
    parser.add_argument("--tied-tol", type=float, default=1e-8)
    args = parser.parse_args()

    frame = load_predictions(args.predictions, extra_columns=["fused_risk"])
    frame = add_standardized_reference_risks(
        frame, already_standardized=args.risks_already_standardized
    )
    result = run_pair_audit(frame, tied_tol=args.tied_tol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("partition", "metrics", "decomposition", "fold_rescue", "rescue_summary"):
        result[key].to_csv(args.output_dir / (key + ".csv"), index=False, float_format="%.10g")
    with (args.output_dir / "checks.json").open("w", encoding="utf-8") as handle:
        json.dump(result["checks"], handle, indent=2)
    print("Pair audit {}: {}".format(result["checks"]["status"], args.output_dir))


if __name__ == "__main__":
    main()
