"""Extract the Sepsyd XGBoost 0.90 pickle into legacy model binary IO.

Run this script only in an isolated environment that has xgboost==0.90.  The
resulting intermediate binary must then be opened and saved as JSON by the
XGBoost 1.7 bridge environment. Use ``convert_legacy_sepsyd_model.sh`` to run
both isolated stages.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import xgboost as xgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Legacy XGBoost 0.90 pickle")
    parser.add_argument("destination", type=Path, help="Intermediate XGBoost binary output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not xgb.__version__.startswith("0.90"):
        raise RuntimeError(
            "Legacy conversion must run with xgboost==0.90; "
            f"found {xgb.__version__}"
        )
    with args.source.open("rb") as handle:
        booster = pickle.load(handle)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(args.destination))
    print(f"Converted {args.source} -> {args.destination}")


if __name__ == "__main__":
    main()
