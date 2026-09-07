from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipelines.processing.feature_builder import add_lookback


def test_lookback_never_crosses_patient_boundary() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p2"],
            "HR": [10.0, 11.0, 20.0, 21.0],
        }
    )

    matrix, columns = add_lookback(frame, ["HR"], hours=2, padding=-1.0)

    assert columns == ["HR__t-0", "HR__t-1"]
    np.testing.assert_array_equal(
        matrix,
        np.array([[10, -1], [11, 10], [20, -1], [21, 20]], dtype=np.float32),
    )

