"""CLI for offline prediction testing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.inference.io import read_patient_file
from src.inference.loader import ModelLoader, ModelLoadError
from src.inference.logger import log_prediction
from src.inference.predictor import PredictionError, run_prediction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run sepsis prediction on a patient file.")
    parser.add_argument("--input", required=True, help="Path to .psv or .csv patient file")
    parser.add_argument("--output", help="Optional path to save results as CSV")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    loader = ModelLoader()
    try:
        df, patient_id = read_patient_file(input_path)
        results, validation, artifact = run_prediction(df, patient_id, loader)

        if not validation.is_valid:
            print("Validation failed:", file=sys.stderr)
            for err in validation.errors:
                print(f"  - {err}", file=sys.stderr)
            log_prediction(
                input_file=input_path.name,
                patient_id=patient_id,
                artifact=None,
                results=None,
                status="validation_failed",
                error="; ".join(validation.errors),
            )
            return 1

        print(results.to_string(index=False))

        if args.output:
            results.to_csv(args.output, index=False)
            print(f"\nSaved to {args.output}")

        log_prediction(
            input_file=input_path.name,
            patient_id=patient_id,
            artifact=artifact,
            results=results,
            status="success",
        )
        return 0

    except (ModelLoadError, PredictionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        log_prediction(
            input_file=input_path.name,
            patient_id=input_path.stem,
            artifact=None,
            results=None,
            status="prediction_failed",
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
