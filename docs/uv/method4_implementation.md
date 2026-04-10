# Method4 实现文档

## 1. 方法定位

`method4_jacobian_injective` 是当前 UV 主线中的非线性 refinement 路线：

1. 先运行 Method2 获取可用初值和内部状态；
2. 在此基础上做可逆性约束下的非线性优化；
3. 若 violation 超阈值，可安全回退 Method2 结果。

实现入口：

1. [run_method4_jacobian_injective](/home/kyrie/FaithC/src/faithc_infra/services/uv/method4_pipeline.py:476)
2. [UVProjector._map_method4_jacobian_injective](/home/kyrie/FaithC/src/faithc_infra/services/uv_projector.py:340)

## 2. 与 Method2 的关系

Method4 不独立构建对应，它依赖 Method2 产出的 `Method2InternalState`：

1. `solve_mesh`
2. `mapped_uv_init`
3. 面目标 Jacobian 及权重
4. 面几何伪逆 `face_geom_pinv`
5. 采样点重心信息与目标 UV
6. 锚点集合与锚点 UV

所以 Method4 的稳定性高度依赖 Method2 初始化质量。

## 3. 非线性能量设计

当前实现能量由以下项组成：

1. Jacobian 数据项：`||J_now - J_target||^2`
2. 邻边平滑项：`||u_i - u_j||^2`
3. Symmetric Dirichlet 项：`||F||^2 + ||F^{-1}||^2`
4. logdet 项：`-log(det_pos)`（`det_pos` 由 softplus 平滑代理）
5. flip barrier：`relu(det_eps - det)^2`
6. legacy barrier（可选）
7. anchor 软约束

实现：

1. [_run_nonlinear_refine / compute_terms](/home/kyrie/FaithC/src/faithc_infra/services/uv/method4_pipeline.py:240)

说明：

1. `det` 相关项使用 softplus 代理，避免硬 clamp 造成梯度不稳定。
2. SD 项负责拉伸/压缩双向约束，flip 项负责负面积区域修复驱动。

## 4. 关键稳定化机制

### 4.1 预修复（pre-repair）

在进入非线性优化前，先对 `det <= det_eps` 的三角形做局部解析梯度修复：

1. 只做小步局部修正
2. 目标是把初值拉回可行域附近

实现：

1. [_pre_repair_inverted_uv](/home/kyrie/FaithC/src/faithc_infra/services/uv/method4_pipeline.py:29)

### 4.2 同伦（homotopy）引入约束

前若干步逐渐放大 barrier 权重：

1. 初期更偏数据项，减少直接撞边界
2. 后期再强化可逆性约束

实现：

1. [method4_pipeline.py](/home/kyrie/FaithC/src/faithc_infra/services/uv/method4_pipeline.py:313)

### 4.3 线搜索回退

每步后检查：

1. 能量是否上升
2. `det` 最小值是否过低

若不满足，沿更新方向做回溯缩步，避免瞬间进入不可行区域。

### 4.4 局部 patch refine

若最终仍有 violation：

1. 提取坏三角相关顶点子集
2. 仅对子集做几轮 Adam 修复
3. 若 violation 或能量更优则接受

## 5. 回退策略

Method4 默认有“可控回退”：

1. 若 `barrier_violations` 同时超过“数量阈值 + 比例阈值”，并且 `fallback_to_method2_on_violation=true`
2. 则输出 Method2 结果，标记 `uv_solver_stage = m2_fallback_after_m4`

这样保证产线不会因为非线性失败而整体失效。

## 6. 关键配置（建议先调这些）

位于 `uv.method4`：

1. `optimizer`（`lbfgs`/`adam`）
2. `max_iters` / `lr`
3. `sym_dirichlet_weight`
4. `logdet_barrier_weight`
5. `flip_barrier_weight`
6. `det_eps` / `det_softplus_beta`
7. `barrier_homotopy_enabled` / `barrier_homotopy_warmup_iters`
8. `pre_repair_enabled` / `pre_repair_iters` / `pre_repair_step`
9. `fallback_violation_ratio_tol` / `fallback_violation_count_tol`

## 7. 主要诊断字段

优先关注：

1. `uv_solver_stage`
2. `uv_m4_refine_status`
3. `uv_m4_nonlinear_iters`
4. `uv_m4_energy_init` / `uv_m4_energy_final`
5. `uv_m4_energy_symd_final`
6. `uv_m4_energy_logdet_final`
7. `uv_m4_energy_flip_final`
8. `uv_m4_barrier_violations`
9. `uv_m4_barrier_violation_ratio`
10. `uv_m4_pre_repair_initial_violations` / `uv_m4_pre_repair_final_violations`

## 8. 当前工程建议

1. Method4 不建议单独跑，建议始终配套 Method2 稳定初始化。
2. 如果频繁回退到 Method2，先检查对应质量和 Method2 约束数量，再调 Method4 权重。
3. 对 hard 资产，建议保留预修复与同伦，避免一步到位硬 barrier。

