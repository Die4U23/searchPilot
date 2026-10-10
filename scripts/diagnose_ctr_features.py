"""四个 CTR 特征各自对点击的分离程度。不训练模型。

切分和种子与 scripts/train_ctr.py 相同。AUC 把特征原值当作分数。
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from searchpilot.ctr.features import FEATURE_NAMES
from searchpilot.ctr.metrics import univariate_separation


def _train_module():
    path = Path(__file__).with_name("train_ctr.py")
    spec = importlib.util.spec_from_file_location("train_ctr", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def render_markdown(rows: list[dict[str, object]], *, n: int, click_rate: float) -> str:
    lines = [
        "# CTR 特征单变量检查",
        "",
        "数据版本 `d3a904f41240`。种子 `20261009`。test 抽样与 `scripts/train_ctr.py` 相同，"
        f"{n} 行，点击率 {_fmt(click_rate)}。",
        "没有训练模型。AUC 把该特征的原值当作分数。",
        "0.5 表示分不开点击。低于 0.5 表示值越大越不容易点击。",
        "",
        "| 特征 | AUC | 点击时均值 | 未点击时均值 | "
        "特征为 0 的行 | 为 0 时点击率 | 大于 0 时点击率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['feature']} | {_fmt(row['auc'])} | {_fmt(row['mean_when_clicked'])} | "
            f"{_fmt(row['mean_when_unclicked'])} | {int(row['n_feature_zero'])} | "
            f"{_fmt(row['click_rate_when_zero'])} | {_fmt(row['click_rate_when_positive'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def diagnose(data_dir: Path) -> str:
    train_ctr = _train_module()
    loaded = train_ctr.load_split_features(data_dir)
    features = np.asarray(loaded[4], dtype=np.float64)
    labels = np.asarray(loaded[5], dtype=np.float64)
    rows = []
    for index, name in enumerate(FEATURE_NAMES):
        stats = univariate_separation(labels, features[:, index])
        stats["feature"] = name
        rows.append(stats)
    return render_markdown(rows, n=int(labels.size), click_rate=float(labels.mean()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("docs/data/ctr-univariate.md"))
    args = parser.parse_args()
    text = diagnose(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
