from __future__ import annotations

from pathlib import Path

from .config import ScenarioConfig, StrategyConfig
from .data import generate_demo_data
from .pipeline import StorePilotPipeline


def main() -> None:
    output = Path("demo_output")
    output.mkdir(exist_ok=True)
    data = generate_demo_data()
    result = StorePilotPipeline().run(
        data,
        strategy=StrategyConfig.preset("平衡", budget=5000),
        scenario=ScenarioConfig(traffic_change_pct=5, supplier_delay_days=1),
    )
    result.base_forecast.to_csv(output / "forecast.csv", index=False)
    result.recommendations.to_csv(output / "recommendations.csv", index=False)
    result.transfers.to_csv(output / "transfers.csv", index=False)
    result.purchase_orders.to_csv(output / "purchase_orders.csv", index=False)
    (output / "daily_report.md").write_text(result.report, encoding="utf-8")
    print(result.report)
    print(
        f"\n模型 WAPE: {result.metrics.wape:.1%}; "
        f"建议数: {len(result.recommendations)}; "
        f"采购单明细: {len(result.purchase_orders)}"
    )


if __name__ == "__main__":
    main()
