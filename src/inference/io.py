"""Patient file I/O."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.shared.schema import (
    CSV_DELIMITER,
    OPTIONAL_COLUMNS,
    PSV_DELIMITER,
    REQUIRED_COLUMNS,
    SUPPORTED_EXTENSIONS,
)


def _delimiter_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".psv":
        return PSV_DELIMITER
    if suffix == ".csv":
        return CSV_DELIMITER
    raise ValueError(
        f"Unsupported file extension '{suffix}'. "
        f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


def read_patient_file(path: Path | str) -> tuple[pd.DataFrame, str]:
    """Read a .psv or .csv patient file.

    Returns:
        Tuple of (dataframe with feature columns, patient_id from filename stem).
    """
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{path.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    patient_id = path.stem
    delimiter = _delimiter_for_path(path)
    df = pd.read_csv(path, sep=delimiter)

    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, patient_id
