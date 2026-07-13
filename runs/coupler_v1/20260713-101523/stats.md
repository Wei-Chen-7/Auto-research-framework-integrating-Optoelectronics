# 统计检验 — 20260713-101523

主要终点两比例 z 检验（pooled，双侧）+ 按任务聚类的配对 bootstrap 95% CI
（10000 次重采样；聚类单元 = (repeat, task_index)，同一 repeat 内四臂共用
同一任务序列，故为配对）。n=60/臂仅足检出大效应，CI 是关键、p 值作参考。

## R3⁻ vs RAE — the sharpest test (DESIGN §12)
（meta_unguarded − meta_governed）

| 终点 | 差值 [95% CI] | z | p |
|---|---|---|---|
| Success Rate | -0.250 [95% CI -0.417, -0.100] | -2.75 | 0.0060 |
| False Accept Rate | +0.346 [95% CI +0.170, +0.522] | 3.40 | 0.0007 |

## RAE vs R2 — does governance cost capability?
（meta_governed − fixed_loop）

| 终点 | 差值 [95% CI] | z | p |
|---|---|---|---|
| Success Rate | +0.067 [95% CI -0.083, +0.217] | 0.73 | 0.4630 |
| False Accept Rate | -0.025 [95% CI -0.206, +0.162] | -0.26 | 0.7933 |

## R3⁻ vs R2 — is unguarded self-mod harmful?
（meta_unguarded − fixed_loop）

| 终点 | 差值 [95% CI] | z | p |
|---|---|---|---|
| Success Rate | -0.183 [95% CI -0.317, -0.050] | -2.03 | 0.0422 |
| False Accept Rate | +0.322 [95% CI +0.161, +0.488] | 3.07 | 0.0021 |
