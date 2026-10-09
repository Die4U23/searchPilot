"""合成出价下的 eCPM 排序对照（PRD FR-7）。

每个 ``item_id`` 由固定种子映射到 ``(0.05, 1.0]``，``source=synthetic``。
这是相对对照，不是真实广告收益；不做竞价、pacing 或 CVR。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

BID_SOURCE = "synthetic"
BID_SEED = 20261009
BID_LOW_EXCLUSIVE = 0.05
BID_HIGH_INCLUSIVE = 1.0
REQUIRED_COLUMNS: tuple[str, ...] = (
    "impression_id",
    "item_id",
    "clicked",
    "pctr_raw",
    "pctr_calibrated",
)


def synthetic_bid(item_id: str, *, seed: int = BID_SEED) -> float:
    """把 ``item_id`` 确定映射到 ``(0.05, 1.0]``；同一种子下同一物品恒定。"""
    material = f"{seed}\0{item_id}".encode()
    n = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    unit = (n + 1) / float(1 << 64)  # (0, 1]
    return BID_LOW_EXCLUSIVE + (BID_HIGH_INCLUSIVE - BID_LOW_EXCLUSIVE) * unit


def synthetic_bid_map(item_ids: Sequence[str], *, seed: int = BID_SEED) -> dict[str, float]:
    """为去重后的 ``item_id`` 生成合成出价表。"""
    return {item_id: synthetic_bid(item_id, seed=seed) for item_id in dict.fromkeys(item_ids)}


def select_top1(item_ids: Sequence[str], scores: Sequence[float]) -> str:
    """分数最高者；并列时取 ``item_id`` 较小者。"""
    if not item_ids:
        raise ValueError("candidates must be non-empty")
    if len(item_ids) != len(scores):
        raise ValueError("item_ids and scores must have the same length")
    best_id = str(item_ids[0])
    best_score = float(scores[0])
    for item_id, score in zip(item_ids[1:], scores[1:], strict=True):
        candidate = str(item_id)
        value = float(score)
        if value > best_score or (value == best_score and candidate < best_id):
            best_id = candidate
            best_score = value
    return best_id


@dataclass(frozen=True, slots=True)
class EcpmArm:
    proxy_revenue: float
    click_hits: int


@dataclass(frozen=True, slots=True)
class EcpmCompareResult:
    source: str
    bid_seed: int
    impression_count: int
    agreement_rate: float
    raw: EcpmArm
    calibrated: EcpmArm

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "bid_seed": self.bid_seed,
            "impression_count": self.impression_count,
            "agreement_rate": self.agreement_rate,
            "raw": {
                "proxy_revenue": self.raw.proxy_revenue,
                "click_hits": self.raw.click_hits,
            },
            "calibrated": {
                "proxy_revenue": self.calibrated.proxy_revenue,
                "click_hits": self.calibrated.click_hits,
            },
        }


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError("missing columns: " + ", ".join(missing))


def _top1_rows(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["impression_id", score_col, "item_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return ordered.drop_duplicates(subset=["impression_id"], keep="first")


def _arm_metrics(top1: pd.DataFrame) -> EcpmArm:
    if top1.empty:
        return EcpmArm(proxy_revenue=0.0, click_hits=0)
    clicked = top1["clicked"].astype(int)
    return EcpmArm(
        proxy_revenue=float((top1["bid"] * clicked).sum()),
        click_hits=int((clicked == 1).sum()),
    )


def compare_ecpm(
    rows: pd.DataFrame,
    *,
    bids: Mapping[str, float] | None = None,
    seed: int = BID_SEED,
) -> EcpmCompareResult:
    """按曝光分别用 bid×原始 pCTR 与 bid×校准 pCTR 选 Top-1，对照代理收入。

    ``bids`` 缺省时用 :func:`synthetic_bid`。自定义出价仍标记 ``source=synthetic``，
    不得解释为真实广告收益。
    """
    _require_columns(rows)
    work = pd.DataFrame(
        {
            "impression_id": rows["impression_id"].map(str),
            "item_id": rows["item_id"].map(str),
            "clicked": pd.to_numeric(rows["clicked"], errors="raise").fillna(0).astype(int),
            "pctr_raw": pd.to_numeric(rows["pctr_raw"], errors="raise").astype(float),
            "pctr_calibrated": pd.to_numeric(rows["pctr_calibrated"], errors="raise").astype(float),
        }
    )
    unique_ids = list(dict.fromkeys(work["item_id"].tolist()))
    if bids is None:
        bid_map = synthetic_bid_map(unique_ids, seed=seed)
    else:
        missing_ids = [item_id for item_id in unique_ids if item_id not in bids]
        if missing_ids:
            raise ValueError("missing bids for: " + ", ".join(missing_ids))
        bid_map = {item_id: float(bids[item_id]) for item_id in unique_ids}

    work["bid"] = work["item_id"].map(bid_map)
    work["ecpm_raw"] = work["bid"] * work["pctr_raw"]
    work["ecpm_calibrated"] = work["bid"] * work["pctr_calibrated"]

    raw_top = _top1_rows(work, "ecpm_raw")
    cal_top = _top1_rows(work, "ecpm_calibrated")
    n_impressions = int(raw_top.shape[0])
    if n_impressions == 0:
        agreement = 0.0
    else:
        merged = raw_top.merge(cal_top, on="impression_id", suffixes=("_raw", "_cal"))
        agreement = float((merged["item_id_raw"] == merged["item_id_cal"]).mean())

    return EcpmCompareResult(
        source=BID_SOURCE,
        bid_seed=seed,
        impression_count=n_impressions,
        agreement_rate=agreement,
        raw=_arm_metrics(raw_top),
        calibrated=_arm_metrics(cal_top),
    )


def format_ecpm_markdown(result: EcpmCompareResult) -> str:
    """对照表。明确标注合成出价，不是真实广告收益。"""
    lines = [
        "# eCPM 排序对照（合成出价）",
        "",
        f"source: `{result.source}`。合成出价种子 `{result.bid_seed}`，"
        "映射区间 `(0.05, 1.0]`。下表是相对对照，**不是真实广告收益**。",
        "",
        f"- 曝光数：{result.impression_count}",
        f"- Top-1 一致率：{result.agreement_rate:.4f}",
        "",
        "| 排序 | 代理收入（合成 bid × 日志点击） | 命中点击数 |",
        "|---|---:|---:|",
        f"| bid × pCTR raw | {result.raw.proxy_revenue:.6f} | {result.raw.click_hits} |",
        (
            f"| bid × pCTR calibrated | {result.calibrated.proxy_revenue:.6f} | "
            f"{result.calibrated.click_hits} |"
        ),
        "",
        "## 为什么校准可以不改变 pCTR 排序，却改变 eCPM 排序",
        "",
        "温度缩放等校准是对 pCTR 的非线性单调变换，单独按 pCTR 排序时 Top-1 不变。"
        "eCPM = 合成出价 × pCTR；出价因物品而异，单调变换之后 Top-1 可以改变。"
        "本实验不做竞价、预算 pacing 或 CVR，结论不得表述为真实广告收益。",
        "",
    ]
    return "\n".join(lines)
