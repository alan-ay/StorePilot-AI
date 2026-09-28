"""Run a reproducible forecast benchmark: python -m storepilot.evaluate."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .data import DEMO_CATALOG_VERSION, RetailData, generate_demo_data
from .evaluation import EvaluationConfig, evaluate_forecasts, evaluation_files
from .i18n import ValidationError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sales", type=Path)
    parser.add_argument("--products", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation_output"))
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 7, 30])
    parser.add_argument("--origins", type=int, default=3)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--min-train-days", type=int, default=70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-days", type=int, default=210)
    parser.add_argument("--demo-end-date", type=date.fromisoformat, default=date(2026, 9, 18))
    args = parser.parse_args(argv)
    paths = (args.sales, args.products, args.inventory)
    if any(path is not None for path in paths) and not all(path is not None for path in paths):
        parser.error("Provide --sales, --products and --inventory together.")
    config = EvaluationConfig(
        horizons=tuple(args.horizons),
        origins=args.origins,
        step_days=args.step_days,
        min_train_days=args.min_train_days,
        seed=args.seed,
    )
    try:
        config.validate()
        if args.sales is not None:
            data = RetailData(
                *[
                    pd.read_csv(path, dtype={"store_id": str, "sku": str, "supplier_id": str})
                    for path in paths
                ]
            )
            source = "门店数据"
        else:
            if args.demo_days < 70:
                parser.error("--demo-days must be at least 70.")
            # The demo generator's date is an exclusive boundary; the CLI accepts
            # the last observed sales date, matching the benchmark manifest.
            data = generate_demo_data(
                args.demo_days, args.seed, args.demo_end_date + timedelta(days=1)
            )
            source = "演示数据"
        result = evaluate_forecasts(data, config)
    except (OSError, ValueError) as exc:
        parser.error(exc.render(args.language) if isinstance(exc, ValidationError) else str(exc))
    if source == "演示数据":
        result.manifest["demo_generation"] = {
            "catalog_version": DEMO_CATALOG_VERSION,
            "days": args.demo_days,
            "end_date": args.demo_end_date.isoformat(),
            "seed": args.seed,
        }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, content in evaluation_files(result, source, args.language).items():
        (args.output / name).write_bytes(content)
    print(f"Evaluation saved to {args.output.resolve() / 'report.md'}")


if __name__ == "__main__":
    main()
