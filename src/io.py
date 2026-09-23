"""Input validation and outer-training normalization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_COLUMNS = (
    "patient_id",
    "time",
    "event",
    "fold",
    "pathology_risk",
    "molecular_risk",
)
TRAIN_STAT_COLUMNS = (
    "pathology_train_mean",
    "pathology_train_std",
    "molecular_train_mean",
    "molecular_train_std",
)


def load_predictions(path: str | Path, extra_columns: Iterable[str] = ()) -> pd.DataFrame:
    """Load one held-out prediction row per patient and official fold."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"patient_id": str})
    required = set(BASE_COLUMNS) | set(extra_columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Prediction CSV is missing columns: {}".format(missing))
    if frame.empty:
        raise ValueError("Prediction CSV is empty")
    if frame.duplicated(["fold", "patient_id"]).any():
        raise ValueError("Duplicate fold/patient_id rows found")
    if frame["patient_id"].duplicated().any():
        raise ValueError("A held-out patient appears in more than one fold")
    if not frame["event"].isin([0, 1, False, True]).all():
        raise ValueError("event must contain only 0/1")
    frame = frame.copy()
    frame["fold"] = frame["fold"].astype(int)
    frame["event"] = frame["event"].astype(int)
    numeric_columns = ["time", "pathology_risk", "molecular_risk", *extra_columns]
    numeric_columns = list(dict.fromkeys(numeric_columns))
    numeric = frame[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("time and risk columns must be finite")
    if (frame["time"].to_numpy(dtype=float) < 0).any():
        raise ValueError("time must be non-negative")
    if frame.groupby("fold").size().min() < 2:
        raise ValueError("Every fold needs at least two patients")
    return frame.sort_values(["fold", "patient_id"]).reset_index(drop=True)


def _validate_fold_constant(frame: pd.DataFrame, column: str) -> None:
    counts = frame.groupby("fold")[column].nunique(dropna=False)
    if not (counts == 1).all():
        raise ValueError("{} must be constant within each fold".format(column))


def add_standardized_reference_risks(
    frame: pd.DataFrame,
    *,
    already_standardized: bool = False,
) -> pd.DataFrame:
    """Add internal ``_pathology_z`` and ``_molecular_z`` columns.

    Priority is: explicit ``*_risk_z`` columns, outer-training mean/std columns,
    then raw risks only when ``already_standardized`` is explicitly requested.
    """

    result = frame.copy()
    z_columns = {"pathology_risk_z", "molecular_risk_z"}
    if z_columns.issubset(result.columns):
        result["_pathology_z"] = result["pathology_risk_z"].astype(float)
        result["_molecular_z"] = result["molecular_risk_z"].astype(float)
    elif set(TRAIN_STAT_COLUMNS).issubset(result.columns):
        for column in TRAIN_STAT_COLUMNS:
            _validate_fold_constant(result, column)
        if (result["pathology_train_std"].astype(float) <= 0).any():
            raise ValueError("pathology_train_std must be positive")
        if (result["molecular_train_std"].astype(float) <= 0).any():
            raise ValueError("molecular_train_std must be positive")
        result["_pathology_z"] = (
            result["pathology_risk"].astype(float)
            - result["pathology_train_mean"].astype(float)
        ) / result["pathology_train_std"].astype(float)
        result["_molecular_z"] = (
            result["molecular_risk"].astype(float)
            - result["molecular_train_mean"].astype(float)
        ) / result["molecular_train_std"].astype(float)
    elif already_standardized:
        result["_pathology_z"] = result["pathology_risk"].astype(float)
        result["_molecular_z"] = result["molecular_risk"].astype(float)
    else:
        raise ValueError(
            "Simple Fusion requires pathology_risk_z and molecular_risk_z, "
            "or the four outer-training mean/std columns. Use "
            "--risks-already-standardized only when the two risk columns are "
            "already standardized with outer-training statistics."
        )
    values = result[["_pathology_z", "_molecular_z"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Standardized reference risks contain NaN/Inf")
    result["simple_fusion_risk"] = 0.5 * (
        result["_pathology_z"] + result["_molecular_z"]
    )
    return result

