# CTR 训练与校准

数据版本 `d3a904f41240`。种子 `20261009`。train / val / test 行数 200000 / 50000 / 50000。
dev 按 `shown_at` 中位数切开，前一半是 val，后一半是 test。校准器只在 val 上拟合。
物品点击只统计 train 且 clicked=1。用户历史来自 `user_history.parquet`。
没有位置特征，CTR 受位置偏差影响，这是已知限制。

- model_version: `ctr-cf4a1955`
- calibrator_version: `temperature-846cd6c1`
- temperature: 4.0538

| 模型 | AUC | LogLoss | ECE |
|---|---:|---:|---:|
| LR | 0.5330 | 0.5000 | 0.3437 |
| MLP raw | 0.4886 | 0.5504 | 0.0376 |
| MLP temperature | 0.4886 | 0.1747 | 0.0155 |

## 分组 ECE（MLP）

- 校准前：`{'history_len_0': 0.038072128270252296, 'history_len_gt_0': 0.034458637414200376}`
- 温度缩放后：`{'history_len_0': 0.0149617618371046, 'history_len_gt_0': 0.05376269840919197}`

## 校准曲线（温度缩放，test）

| bin | count | confidence | accuracy |
|---|---:|---:|---:|
| 0.0-0.1 | 46772 | 0.0292 | 0.0380 |
| 0.1-0.2 | 2652 | 0.1264 | 0.0351 |
| 0.2-0.3 | 516 | 0.2443 | 0.0407 |
| 0.3-0.4 | 60 | 0.3229 | 0.0167 |
