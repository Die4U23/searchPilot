"""抽样训练 CTR：LR 与 MLP，校准器只拟合 val。

train 曝光随机抽样。dev 按接近行数均分的完整 shown_at 时间组切成 val / test，不拆分曝光。
种子 20261009。不用 test 选择超参。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from searchpilot.ctr.artifacts import export_bundle
from searchpilot.ctr.calibrate import (
    apply_isotonic,
    apply_platt,
    apply_temperature,
    fit_isotonic,
    fit_platt,
    fit_temperature,
)
from searchpilot.ctr.features import (
    FEATURE_NAMES,
    apply_standardizer,
    fit_standardizer,
    rows_to_matrix,
)
from searchpilot.ctr.metrics import binary_auc, binary_log_loss, expected_calibration_error
from searchpilot.ctr.model import (
    EPOCHS,
    LEARNING_RATE,
    SEED,
    LogisticModel,
    MlpModel,
    export_linear,
    export_mlp,
    predict_logits,
    train_binary,
)
from searchpilot.ctr.split import split_dev_by_time
from searchpilot.ctr.temporal_features import FEATURE_POLICY, training_features

TRAIN_N = 200_000
VAL_N = 50_000
TEST_N = 50_000
DATA_VERSION = "d3a904f41240"


def _sample_split(frame: pd.DataFrame, size: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(frame) <= size:
        return frame.reset_index(drop=True)
    picked = rng.choice(len(frame), size=size, replace=False)
    return frame.iloc[np.sort(picked)].reset_index(drop=True)


def _json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metrics_block(name: str, labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    ece, curve = expected_calibration_error(labels, probabilities)
    return {
        "name": name,
        "auc": binary_auc(labels, probabilities),
        "log_loss": binary_log_loss(labels, probabilities),
        "ece": ece,
        "curve": curve,
    }


def _slice_metrics(
    labels: np.ndarray, probabilities: np.ndarray, mask: np.ndarray
) -> dict[str, object]:
    count = int(mask.sum())
    if count == 0:
        return {"n": 0, "auc": None, "log_loss": None, "ece": None}
    y = labels[mask]
    p = probabilities[mask]
    ece, _curve = expected_calibration_error(y, p)
    return {
        "n": count,
        "auc": binary_auc(y, p),
        "log_loss": binary_log_loss(y, p),
        "ece": ece,
    }


def _group_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    history: np.ndarray,
    popularity: np.ndarray,
) -> dict[str, dict[str, object]]:
    """历史长度与 train 期物品点击分桶。分桶只用训练期计数。"""
    masks = {
        "history_len_0": history <= 0,
        "history_len_gt_0": history > 0,
        "item_clicks_0": popularity <= 0,
        "item_clicks_gt_0": popularity > 0,
    }
    return {key: _slice_metrics(labels, probabilities, mask) for key, mask in masks.items()}


def load_split_features(data_dir: Path) -> tuple[object, ...]:
    """与训练同一套种子和切分。返回三个 split 的未标准化特征，以及原始抽样表。"""
    root = data_dir / "processed" / DATA_VERSION
    impressions = pq.read_table(
        root / "impressions.parquet",
        columns=["impression_id", "user_id", "item_id", "clicked", "split", "shown_at"],
    ).to_pandas()
    items = pq.read_table(
        root / "items.parquet", columns=["item_id", "title", "category"]
    ).to_pandas()

    item_category = dict(
        zip(items["item_id"].astype(str), items["category"].astype(str), strict=True)
    )
    item_title_chars = {
        str(item_id): len(str(title))
        for item_id, title in zip(items["item_id"], items["title"], strict=True)
    }
    train = impressions.loc[impressions["split"] == "train"].reset_index(drop=True).copy()
    train["ctr_row_id"] = np.arange(len(train))
    dev = impressions.loc[impressions["split"] == "dev"]
    train_times = pd.to_datetime(train["shown_at"], utc=True, errors="raise")
    dev_times = pd.to_datetime(dev["shown_at"], utc=True, errors="raise")
    if (
        train.empty
        or dev.empty
        or train_times.isna().any()
        or dev_times.isna().any()
        or train_times.max() >= dev_times.min()
    ):
        raise ValueError("all training events must precede the dev time window")
    rng = np.random.default_rng(SEED)
    train_rows = _sample_split(train, TRAIN_N, rng)
    x_train, tables = training_features(
        train,
        train_rows["ctr_row_id"].tolist(),
        item_category=item_category,
        item_title_chars=item_title_chars,
    )
    y_train = train_rows["clicked"].to_numpy(dtype=np.int64)
    item_clicks = tables["item_clicks"]
    user_history_len = tables["user_history_len"]
    user_category_counts = tables["user_category_counts"]
    val_partition, test_partition = split_dev_by_time(dev)
    val_rows = _sample_split(val_partition, VAL_N, rng)
    test_rows = _sample_split(test_partition, TEST_N, rng)

    def matrix_of(
        frame: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        users = frame["user_id"].astype(str)
        item_ids = frame["item_id"].astype(str)
        pairs = list(zip(users, item_ids, strict=True))
        raw = rows_to_matrix(
            pairs,
            user_history_len=user_history_len,
            user_category_counts=user_category_counts,
            item_clicks=item_clicks,
            item_category=item_category,
            item_title_chars=item_title_chars,
        )
        history_len = np.array([user_history_len.get(user, 0) for user in users], dtype=np.int64)
        popularity = np.array([int(item_clicks.get(item, 0)) for item in item_ids], dtype=np.int64)
        return raw, frame["clicked"].to_numpy(dtype=np.int64), history_len, popularity

    x_val, y_val, _h_val, _p_val = matrix_of(val_rows)
    x_test, y_test, h_test, p_test = matrix_of(test_rows)
    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        h_test,
        p_test,
        train_rows,
        val_rows,
        test_rows,
        item_clicks,
        item_category,
        item_title_chars,
        user_history_len,
        user_category_counts,
    )


def train(data_dir: Path, artifact_dir: Path) -> dict[str, object]:
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        h_test,
        p_test,
        train_rows,
        val_rows,
        test_rows,
        item_clicks,
        item_category,
        item_title_chars,
        user_history_len,
        user_category_counts,
    ) = load_split_features(data_dir)
    mean, std = fit_standardizer(x_train)
    train_x = apply_standardizer(x_train, mean, std)
    val_x = apply_standardizer(x_val, mean, std)
    test_x = apply_standardizer(x_test, mean, std)

    width = len(FEATURE_NAMES)
    lr = train_binary(LogisticModel(width), train_x, y_train)
    mlp = train_binary(MlpModel(width), train_x, y_train, seed=SEED + 1)
    lr_w = export_linear(lr)
    mlp_w = export_mlp(mlp)
    val_logits = predict_logits(mlp, val_x)
    test_logits = predict_logits(mlp, test_x)
    lr_test = predict_logits(lr, test_x)

    temperature = fit_temperature(val_logits, y_val)
    platt_coef, platt_intercept = fit_platt(val_logits, y_val)
    iso_x, iso_y = fit_isotonic(1.0 / (1.0 + np.exp(-np.clip(val_logits, -60, 60))), y_val)
    raw_test = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -60, 60)))
    cal_test = apply_temperature(test_logits, temperature)
    platt_test = apply_platt(test_logits, platt_coef, platt_intercept)
    iso_test = apply_isotonic(raw_test, iso_x, iso_y)
    lr_prob = 1.0 / (1.0 + np.exp(-np.clip(lr_test, -60, 60)))

    out = artifact_dir / "ctr"
    model_version, calibrator_version = export_bundle(
        out,
        meta={
            "feature_names": list(FEATURE_NAMES),
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "data_version": DATA_VERSION,
            "feature_policy": FEATURE_POLICY,
            "seed": SEED,
            "train_rows": int(len(train_rows)),
            "val_rows": int(len(val_rows)),
            "test_rows": int(len(test_rows)),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
        },
        lr={"weight": lr_w.weight, "bias": lr_w.bias},
        mlp={"w1": mlp_w.w1, "b1": mlp_w.b1, "w2": mlp_w.w2, "b2": mlp_w.b2},
        temperature=temperature,
        feature_tables={
            "item_clicks": {k: int(v) for k, v in item_clicks.items()},
            "item_category": item_category,
            "item_title_chars": {k: int(v) for k, v in item_title_chars.items()},
            "user_history_len": {k: int(v) for k, v in user_history_len.items()},
            "user_category_counts": user_category_counts,
        },
    )
    _json(
        out / "calibrators.json",
        {
            "temperature": temperature,
            "platt": {"coef": platt_coef, "intercept": platt_intercept},
            "isotonic": {"x": iso_x, "y": iso_y},
        },
    )
    scores = test_rows[["impression_id", "user_id", "item_id", "clicked"]].copy()
    scores["clicked"] = scores["clicked"].astype(int)
    scores["pctr_raw"] = raw_test
    scores["pctr_calibrated"] = cal_test
    scores["split"] = "test"
    scores.to_parquet(out / "test_scores.parquet", index=False)

    raw_block = _metrics_block("mlp_raw", y_test, raw_test)
    cal_block = _metrics_block("mlp_temperature", y_test, cal_test)
    platt_block = _metrics_block("mlp_platt", y_test, platt_test)
    iso_block = _metrics_block("mlp_isotonic", y_test, iso_test)
    lr_block = _metrics_block("lr", y_test, lr_prob)
    report = {
        "feature_policy": FEATURE_POLICY,
        "model_version": model_version,
        "calibrator_version": calibrator_version,
        "temperature": temperature,
        "rows": {
            "train": int(len(train_rows)),
            "val": int(len(val_rows)),
            "test": int(len(test_rows)),
        },
        "lr": lr_block,
        "mlp_raw": raw_block,
        "mlp_temperature": cal_block,
        "mlp_platt": platt_block,
        "mlp_isotonic": iso_block,
        "grouped_temperature": _group_metrics(y_test, cal_test, h_test, p_test),
        "grouped_ece": {
            "raw": _group_metrics(y_test, raw_test, h_test, p_test),
            "temperature": _group_metrics(y_test, cal_test, h_test, p_test),
            "platt": _group_metrics(y_test, platt_test, h_test, p_test),
            "isotonic": _group_metrics(y_test, iso_test, h_test, p_test),
        },
    }
    _json(out / "metrics.json", report)
    return report


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


_GROUP_TITLES = (
    ("history_len_0", "历史长度 0"),
    ("history_len_gt_0", "历史长度 > 0"),
    ("item_clicks_0", "train 点击 0"),
    ("item_clicks_gt_0", "train 点击 > 0"),
)


def _group_lines(groups: object) -> list[str]:
    assert isinstance(groups, dict)
    lines = [
        "| 分组 | n | AUC | LogLoss | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, title in _GROUP_TITLES:
        block = groups[key]
        assert isinstance(block, dict)
        lines.append(
            f"| {title} | {int(block['n'])} | {_fmt(block['auc'])} | "
            f"{_fmt(block['log_loss'])} | {_fmt(block['ece'])} |"
        )
    return lines


def _calibrator_ece_lines(groups: object) -> list[str]:
    assert isinstance(groups, dict)
    lines = [
        "| 分组 | n | raw | temperature | Platt | isotonic |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, title in _GROUP_TITLES:
        cells = []
        count = None
        for name in ("raw", "temperature", "platt", "isotonic"):
            block = groups[name][key]
            assert isinstance(block, dict)
            count = int(block["n"])
            cells.append(_fmt(block["ece"]))
        lines.append(f"| {title} | {count} | {' | '.join(cells)} |")
    return lines


def write_report(report: dict[str, object], path: Path) -> None:
    rows = report["rows"]
    assert isinstance(rows, dict)
    lr = report["lr"]
    raw = report["mlp_raw"]
    cal = report["mlp_temperature"]
    platt = report["mlp_platt"]
    isotonic = report["mlp_isotonic"]
    assert (
        isinstance(lr, dict)
        and isinstance(raw, dict)
        and isinstance(cal, dict)
        and isinstance(platt, dict)
        and isinstance(isotonic, dict)
    )
    lines = [
        "# CTR 训练与校准",
        "",
        f"数据版本 `{DATA_VERSION}`。种子 `{SEED}`。train / val / test 行数 "
        f"{rows['train']} / {rows['val']} / {rows['test']}。",
        "dev 按接近行数均分的完整 `shown_at` 时间组切开，不拆分曝光；"
        "先 val 后 test，再各自抽样。校准器只在 val 上拟合。",
        "训练特征仅使用严格早于当前曝光时间的 train 正反馈；同一时间戳共享冻结前缀。",
        "用户历史从观测窗口起点累积，不回填 `user_history.parquet` 的最终历史。",
        "val/test 与线上使用训练窗口结束时冻结的计数，不使用 dev 标签。",
        "此协议改变了历史特征定义，旧报告指标不能直接作为本轮结果。",
        "没有位置特征，CTR 受位置偏差影响，这是已知限制。",
        "线上默认校准器仍是温度缩放。Platt 与等渗回归只在这份对照里出现。",
        "",
        f"- model_version: `{report['model_version']}`",
        f"- calibrator_version: `{report['calibrator_version']}`",
        f"- temperature: {float(report['temperature']):.4f}",
        "",
        "| 模型 | AUC | LogLoss | ECE |",
        "|---|---:|---:|---:|",
        f"| LR | {_fmt(lr['auc'])} | {_fmt(lr['log_loss'])} | {_fmt(lr['ece'])} |",
        f"| MLP raw | {_fmt(raw['auc'])} | {_fmt(raw['log_loss'])} | {_fmt(raw['ece'])} |",
        f"| MLP temperature | {_fmt(cal['auc'])} | {_fmt(cal['log_loss'])} | {_fmt(cal['ece'])} |",
        f"| MLP Platt | {_fmt(platt['auc'])} | {_fmt(platt['log_loss'])} | {_fmt(platt['ece'])} |",
        "| MLP isotonic | "
        f"{_fmt(isotonic['auc'])} | {_fmt(isotonic['log_loss'])} | {_fmt(isotonic['ece'])} |",
        "",
        "## 分组指标（温度缩放，test）",
        "",
        *_group_lines(report["grouped_temperature"]),
        "",
        "## 分组 ECE（三种校准器，test）",
        "",
        *_calibrator_ece_lines(report["grouped_ece"]),
        "",
        "## 校准曲线（温度缩放，test）",
        "",
        "| bin | count | confidence | accuracy |",
        "|---|---:|---:|---:|",
    ]
    curve = cal["curve"]
    assert isinstance(curve, list)
    for point in curve:
        assert isinstance(point, dict)
        lines.append(
            f"| {point['bin_left']:.1f}-{point['bin_right']:.1f} | {int(point['count'])} | "
            f"{point['confidence']:.4f} | {point['accuracy']:.4f} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CTR models and val-only calibrators")
    parser.add_argument(
        "--data-dir", type=Path, default=Path(os.environ.get("SEARCHPILOT_DATA_DIR", "./data"))
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(os.environ.get("SEARCHPILOT_ARTIFACT_DIR", "./artifacts")),
    )
    parser.add_argument("--report", type=Path, default=Path("docs/data/ctr-report.md"))
    args = parser.parse_args(argv)
    report = train(args.data_dir, args.artifact_dir)
    write_report(report, args.report)
    print(f"model_version={report['model_version']}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
