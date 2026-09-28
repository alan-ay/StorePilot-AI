"""Chronological, recursive forecast comparisons against observed sales."""

from __future__ import annotations

import io
import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from numbers import Integral
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from .data import RetailData, validate_data
from .forecasting import AdaptiveDemandForecaster
from .i18n import ValidationError, translate
from .workbench import dataset_key, download_csv

MODEL_LABELS = {
    "seasonal_naive": "重复上周销量",
    "moving_average_28": "近28天平均销量",
    "gradient_boosting": "梯度提升模型",
}
SCORE_COLUMNS = {
    "model": "模型",
    "horizon_days": "预测窗口（天）",
    "mae": "每日 MAE",
    "wape_pct": "WAPE (%)",
    "bias": "每日偏差",
    "window_mae": "窗口总量 MAE",
    "non_stockout_mae": "非缺货日 MAE",
    "scored_rows": "评估记录数",
    "stockout_rows": "缺货记录数",
}


@dataclass(frozen=True)
class EvaluationConfig:
    horizons: tuple[int, ...] = (1, 7, 30)
    origins: int = 3
    step_days: int = 7
    min_train_days: int = 70
    seed: int = 42

    def validate(self):
        def integer(value, low, high):
            return (
                isinstance(value, Integral) and not isinstance(value, bool) and low <= value <= high
            )

        if (
            not self.horizons
            or any(not integer(h, 1, 90) for h in self.horizons)
            or len(set(self.horizons)) != len(self.horizons)
        ):
            raise ValidationError("评估窗口需为不重复的 1 至 90 天整数")
        if not integer(self.origins, 1, 10) or not integer(self.step_days, 1, 90):
            raise ValidationError("评估需使用 1 至 10 个截止日，间隔为 1 至 90 天整数")
        if not integer(self.min_train_days, 70, 3650) or not integer(self.seed, 0, 2**32 - 1):
            raise ValidationError("评估训练期至少为 70 天，随机种子需为有效非负整数")


@dataclass
class EvaluationResult:
    predictions: pd.DataFrame
    scores: pd.DataFrame
    sku_scores: pd.DataFrame
    fold_scores: pd.DataFrame
    excluded: pd.DataFrame
    manifest: dict


def score_predictions(predictions, horizons, group_columns=("model",)):
    """Pool daily errors within each prefix window, with one weight per forecast row."""
    rows = []
    for horizon in sorted(horizons):
        window = predictions[predictions.lead_day <= horizon]
        for keys, group in window.groupby(list(group_columns), sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            error = group.predicted_units - group.actual_units
            total = float(group.actual_units.sum())
            totals = group.groupby(["origin", "store_id", "sku"])[
                ["actual_units", "predicted_units"]
            ].sum()
            non_stockout = error[group.stockout.eq(0)]
            rows.append(
                dict(
                    zip(group_columns, keys, strict=True),
                    horizon_days=horizon,
                    mae=float(error.abs().mean()),
                    wape=float(error.abs().sum() / total) if total else float("nan"),
                    bias=float(error.mean()),
                    window_mae=float((totals.predicted_units - totals.actual_units).abs().mean()),
                    non_stockout_mae=float(non_stockout.abs().mean()),
                    scored_rows=len(group),
                    stockout_rows=int(group.stockout.sum()),
                )
            )
    return pd.DataFrame(rows)


def evaluate_forecasts(
    data: RetailData, config: EvaluationConfig | None = None
) -> EvaluationResult:
    config = config or EvaluationConfig()
    config.validate()
    data = validate_data(data)
    sales = data.sales
    horizon = max(config.horizons)
    latest = sales.date.max() - pd.Timedelta(days=horizon)
    cutoffs = [
        latest - pd.Timedelta(days=i * config.step_days) for i in reversed(range(config.origins))
    ]
    eligible, excluded = [], []
    for (store, sku), group in sales.groupby(["store_id", "sku"], sort=True):
        if len(group) != (group.date.max() - group.date.min()).days + 1:
            reason = "缺少每日记录；请明确补齐零销量日"
        elif int(group.date.le(cutoffs[0]).sum()) < config.min_train_days:
            reason = "首个截止日前的训练历史不足"
        else:
            eligible.append(group)
            continue
        excluded.append({"store_id": store, "sku": sku, "reason": reason})
    if not eligible:
        required = config.min_train_days + horizon + (config.origins - 1) * config.step_days
        raise ValidationError("没有可评估的商品；当前设置至少需要 {0} 天连续每日记录", required)
    sales = pd.concat(eligible, ignore_index=True)
    predictions = []
    fold_details = []
    for cutoff in cutoffs:
        # No future sales, prices, promotion flags, or stockout corrections enter fitting.
        train = sales[sales.date <= cutoff].copy()
        model = AdaptiveDemandForecaster(random_state=config.seed).fit(train)
        learned = model.predict(horizon)[["date", "store_id", "sku", "predicted_units"]]
        learned["model"] = "gradient_boosting"
        baseline_rows = []
        for (store, sku), group in train.groupby(["store_id", "sku"], sort=True):
            units = group.sort_values("date").units.to_numpy()
            for day in range(1, horizon + 1):
                common = {
                    "date": cutoff + pd.Timedelta(days=day),
                    "store_id": store,
                    "sku": sku,
                }
                baseline_rows.extend(
                    [
                        dict(
                            common,
                            model="seasonal_naive",
                            predicted_units=float(units[-7:][(day - 1) % 7]),
                        ),
                        dict(
                            common,
                            model="moving_average_28",
                            predicted_units=float(units[-28:].mean()),
                        ),
                    ]
                )
        forecast = pd.concat([learned, pd.DataFrame(baseline_rows)], ignore_index=True)
        actual = sales[(sales.date > cutoff) & (sales.date <= cutoff + pd.Timedelta(days=horizon))]
        scored = forecast.merge(
            actual[["date", "store_id", "sku", "units", "stockout"]],
            on=["date", "store_id", "sku"],
            how="left",
            validate="many_to_one",
        ).rename(columns={"units": "actual_units"})
        if scored.actual_units.isna().any():
            raise ValidationError("评估窗口内缺少实际销量，不能将缺失记录当作零销量")
        scored["origin"] = cutoff
        scored["lead_day"] = (scored.date - cutoff).dt.days
        predictions.append(scored)
        fold_details.append(
            {
                "origin": cutoff.date().isoformat(),
                "training_rows": len(train),
                "training_start": train.date.min().date().isoformat(),
                "test_end": (cutoff + pd.Timedelta(days=horizon)).date().isoformat(),
            }
        )
    predictions = (
        pd.concat(predictions, ignore_index=True)
        .sort_values(["origin", "model", "store_id", "sku", "date"])
        .reset_index(drop=True)
    )
    return EvaluationResult(
        predictions=predictions,
        scores=score_predictions(predictions, config.horizons),
        sku_scores=score_predictions(predictions, config.horizons, ("store_id", "sku", "model")),
        fold_scores=score_predictions(predictions, config.horizons, ("origin", "model")),
        excluded=pd.DataFrame(excluded, columns=["store_id", "sku", "reason"]),
        manifest={
            "protocol": "rolling-origin-v1",
            "config": asdict(config),
            "dataset_id": dataset_key(data, "forecast-evaluation-v1"),
            "sales_start": data.sales.date.min().date().isoformat(),
            "sales_end": data.sales.date.max().date().isoformat(),
            "evaluated_series": len(eligible),
            "excluded_series": len(excluded),
            "overlapping_windows": config.origins > 1 and config.step_days < horizon,
            "folds": fold_details,
            "model_parameters": model.model.get_params(),
            "versions": {
                "python": platform.python_version(),
                **{
                    name: version(name) for name in ("numpy", "pandas", "scikit-learn", "streamlit")
                },
            },
        },
    )


def score_table(scores, language="zh"):
    frame = scores.copy()
    frame["model"] = frame.model.map(lambda model: translate(MODEL_LABELS[model], language))
    frame["wape_pct"] = frame.wape * 100
    return frame[list(SCORE_COLUMNS)].rename(
        columns={key: translate(label, language) for key, label in SCORE_COLUMNS.items()}
    )


def evaluation_report(result: EvaluationResult, source: str, language="en") -> str:
    def t(message, *values):
        return translate(message, language, *values)

    manifest = result.manifest
    lines = [
        t("# StorePilot 预测评估报告"),
        "",
        t("数据来源：{0}。数据编号：{1}。", t(source), manifest["dataset_id"]),
        t(
            "销售日期：{0} 至 {1}。评估 {2} 条商品序列，排除 {3} 条。",
            manifest["sales_start"],
            manifest["sales_end"],
            manifest["evaluated_series"],
            manifest["excluded_series"],
        ),
        t("历史截止日：{0}。", ", ".join(fold["origin"] for fold in manifest["folds"])),
        "",
        t("## 方法与指标"),
        t(
            "各模型只使用截止日及之前的数据。梯度提升逐日递归预测；两个基线分别重复最后一周和固定使用最近28天平均销量。"
        ),
        t(
            "梯度提升沿用应用的缺货修正训练目标；两个基线使用历史实际销量。未来促销与价格保持应用的默认假设。"
        ),
        t(
            "每个窗口评估截止日后的第1天至第H天；每日 MAE 衡量逐日绝对误差，窗口总量 MAE 衡量每个商品在整个窗口内的销量总量误差。"
        ),
        t("WAPE 为绝对误差之和除以实际销量之和；实际销量全为零时不定义。正偏差表示预测偏高。"),
        t(
            "实际销量直接来自记录，不用估算需求替代。缺货会压低实际销量，因此另列非缺货日 MAE；这不能证明真实需求准确率。"
        ),
        t("历史记录不连续或训练期不足的商品会明确排除。三个模型使用相同的商品与评估日期。"),
        "",
        t("## 对比结果"),
        "",
    ]
    table = score_table(result.scores, language)
    lines.extend(
        [
            "| " + " | ".join(table.columns) + " |",
            "| " + " | ".join(["---"] * len(table.columns)) + " |",
        ]
    )
    for row in table.itertuples(index=False, name=None):
        cells = [
            "—" if pd.isna(value) else f"{value:.3f}" if isinstance(value, float) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", t("## 结果边界")])
    if source == "演示数据":
        lines.append(t("这是模拟演示数据的实验，不代表真实门店的预测效果或经营收益。"))
    if manifest["overlapping_windows"]:
        lines.append(
            t(
                "评估窗口存在重叠；同一销售日可能被多个截止日预测，汇总结果不应视为独立样本的统计显著性证据。"
            )
        )
    lines.extend(
        [
            t("对比不会自动更换应用中的模型，也不会修改备货偏好、库存或采购记录。"),
            t(
                "结果仅衡量历史销量预测误差，不证明库存成本下降。此报告不评估预测区间覆盖率，也未实现自动模型选择。"
            ),
            "",
            t("## 复现信息"),
            t(
                "完整设置、软件版本和训练参数见 manifest.json；逐条预测、截止日指标和商品指标可用于复核。"
            ),
            "",
            "```json",
        ]
    )
    lines.extend([json.dumps(manifest, ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def evaluation_files(result: EvaluationResult, source: str, language="en") -> dict[str, bytes]:
    """Share the same audit artifacts between the dashboard and command line."""
    outputs = {
        "report.md": evaluation_report(result, source, language).encode("utf-8"),
        "manifest.json": json.dumps(
            dict(result.manifest, source=translate(source, "en")), ensure_ascii=False, indent=2
        ).encode("utf-8"),
    }
    for name, frame in {
        "summary": result.scores,
        "predictions": result.predictions,
        "sku_scores": result.sku_scores,
        "fold_scores": result.fold_scores,
        "excluded": result.excluded.assign(
            reason=result.excluded.reason.map(lambda value: translate(value, language))
        ),
    }.items():
        outputs[f"{name}.csv"] = download_csv(frame)
    return outputs


def evaluation_archive(result: EvaluationResult, source: str, language="en") -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in evaluation_files(result, source, language).items():
            archive.writestr(name, content)
    return buffer.getvalue()
