# 统计检验 — 20260713-120554

主要终点两比例 z 检验（pooled，双侧）+ 按任务聚类的配对 bootstrap 95% CI
（10000 次重采样；聚类单元 = (repeat, task_index)，同一 repeat 内四臂共用
同一任务序列，故为配对）。n=60/臂仅足检出大效应，CI 是关键、p 值作参考。

## R3⁻ vs RAE — the sharpest test (DESIGN §12)
（meta_unguarded − meta_governed）

| 终点 | 差值 [95% CI] | z | p |
|---|---|---|---|
| Success Rate | +0.100 [95% CI +0.000, +0.200] | 1.10 | 0.2709 |
| False Accept Rate | -0.053 [95% CI -0.171, +0.063] | -0.56 | 0.5730 |

## RAE vs R2 — does governance cost capability?
（meta_governed − fixed_loop）

| 终点 | 差值 [95% CI] | z | p |
|---|---|---|---|
| Success Rate | -0.083 [95% CI -0.167, +0.000] | -0.92 | 0.3596 |
| False Accept Rate | +0.062 [95% CI -0.036, +0.167] | 0.66 | 0.5114 |

## R3⁻ vs R2 — is unguarded self-mod harmful?
（meta_unguarded − fixed_loop）

| 终点 | 差值 [95% CI] | z | p |
|---|---|---|---|
| Success Rate | +0.017 [95% CI -0.067, +0.100] | 0.19 | 0.8527 |
| False Accept Rate | +0.009 [95% CI -0.100, +0.120] | 0.10 | 0.9191 |
