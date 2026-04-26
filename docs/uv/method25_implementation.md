# Method2.5 实现文档

## 1. 方法定位

`method25_projected_jacobian_injective` 是一条实验性组合路线：

1. 先运行 Method2，拿到对应、线性求解初值和完整内部状态；
2. 不直接沿用 Method2 原始面级目标 Jacobian；
3. 用 Exp13 主线上验证过的“锚点基底 + sample-fit 残差投影”重建一版更干净的目标场；
4. 先对这版目标场做一次 Method2 线性积分，作为新的初始化；
5. 再把这版初始化和目标场交给 Method4 的非线性求解器。

它不是新的默认方法，只是为了回答一个更明确的问题：

1. 如果把“更干净的场”喂给“有 barrier 的求解器”，Stretch 是否会明显改善？
2. 如果仍然不能改善，问题就更可能在场本身，而不是线性积分器。

## 2. 当前默认投影变体

当前实现固定采用 Exp13 的最佳残差 projector 思路：

1. 从采样点直接拟合每个低模面的局部 Jacobian；
2. 用锚点诱导出的底图 Jacobian 作为安全基底；
3. 在残差空间做带 curl 正则的二次投影；
4. 默认配置是 soft gate、`lambda_curl = 20`、`lambda_decay = 1`。

相关实现：

1. [field_projector.py](/home/kyrie/FaithC/src/faithc_infra/services/uv/field_projector.py)
2. [method25_pipeline.py](/home/kyrie/FaithC/src/faithc_infra/services/uv/method25_pipeline.py)

## 3. 与 Method2 / Method4 的关系

Method2.5 不是独立系统，它依赖两边：

1. 上游依赖 Method2 的 correspondence、sample 数据、anchor 数据和 solve mesh；
2. 下游直接复用 Method4 的非线性 refine、barrier、预修复、同伦和回退逻辑。

因此它的工程含义更接近：

1. “Projector V2 + Method4 Solver”
2. 而不是“全新 UV 方法”

## 4. 当前主要输出字段

优先关注：

1. `uv_solver_stage`
2. `uv_m25_enabled`
3. `uv_m25_field_source`
4. `uv_m25_projector_matrix_meta`
5. `uv_m25_projector_solve_meta`
6. `uv_m25_linear_init_solver_meta`
7. `uv_m4_refine_status`
8. `uv_m4_barrier_violations`

## 5. 目前的边界

这条路线当前只回答一个核心问题：

1. “干净场 + 非线性 barrier solver” 是否优于 “干净场 + Method2 线性积分”。

它还没有覆盖：

1. Route B 的显式 solve-space box 约束；
2. Route C 的 topology release；
3. 更一般化的 projector 变体搜索。
