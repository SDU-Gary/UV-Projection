# Method2 实现文档

## 1. 方法定位

`method2_gradient_poisson` 是当前 UV 主线之一，目标是：

1. 从低模采样点到高模建立稳健对应；
2. 将高模 UV 的局部梯度（Jacobian）传递到低模；
3. 通过稀疏线性系统解出低模顶点 UV。

实现入口：

1. [run_method2_gradient_poisson](/home/kyrie/FaithC/src/faithc_infra/services/uv/method2_pipeline.py:682)
2. [UVProjector._map_method2_gradient_poisson](/home/kyrie/FaithC/src/faithc_infra/services/uv_projector.py:320)

## 2. 输入输出

输入：

1. 高模网格（必须含 UV）
2. 低模网格
3. 可选贴图图像（用于梯度加权和颜色重投影指标）
4. `uv` 配置中的 `sample / correspondence / seam / solve / method2` 子项

输出：

1. `mapped_uv`（低模顶点 UV）
2. `method_stats`（求解与诊断字段）
3. `export_payload`（含 halfedge split 后拓扑、质量评估 mesh 等）

## 3. 算法流程

### 3.1 低模采样

使用 [sample_low_mesh](/home/kyrie/FaithC/src/faithc_infra/services/uv/sampling.py) 在低模面上生成采样点，记录：

1. 采样点坐标、法线
2. 所属低模面 id
3. 重心坐标
4. 面积权重

### 3.2 对应查找

调用 [correspond_points_hybrid](/home/kyrie/FaithC/src/faithc_infra/services/uv/correspondence.py)：

1. 双向法线射线 + UDF fallback
2. 法线一致性过滤
3. 输出采样点目标 UV、目标高模面 id、有效掩码、fallback 标记

### 3.3 接缝/岛诊断与拓扑处理

核心逻辑在 [method2_pipeline.py](/home/kyrie/FaithC/src/faithc_infra/services/uv/method2_pipeline.py:581)：

1. 计算高模 UV 岛标签
2. 将低模面映射到高模岛标签并检测冲突面
3. 可选 `halfedge_island` 切缝并复制低模顶点
4. 检测跨缝面（UV span 超阈值）并可从方程中剔除
5. 可选 island guard（默认建议由 `method2.use_island_guard` 控制，当前默认关闭）

### 3.4 面级 Jacobian 估计与稳健聚合

步骤：

1. 先计算高模每个三角面的 UV Jacobian（`2x3`）
2. 按低模面聚合其对应采样点的高模 Jacobian
3. 聚合使用 IRLS + Huber + 分位/MAD 异常抑制

实现：

1. [_compute_high_face_jacobians](/home/kyrie/FaithC/src/faithc_infra/services/uv/method2_pipeline.py:76)
2. [_aggregate_face_target_jacobians](/home/kyrie/FaithC/src/faithc_infra/services/uv/method2_pipeline.py:98)

### 3.5 自适应平滑（细节保留）

当前实现为“面方差驱动的平滑强度”：

1. 每面 Jacobian 协方差迹 `cov_trace` 越大，`alpha_f` 越小；
2. `alpha_f` 投影到边上构造加权 Laplacian；
3. 平滑项从统一 `L` 变为“按局部变化自适应”的 `L_w`。

实现：

1. [_weighted_edge_laplacian_from_face_alpha](/home/kyrie/FaithC/src/faithc_infra/services/uv/method2_pipeline.py:216)

### 3.6 线性系统构建与求解

约束系统：

1. 每个有效低模面 Jacobian 生成 3 条边约束（u/v 分通道）
2. 组成稀疏矩阵 `A`
3. 解 `(A^T A + λL + anchor + ridge) x = A^T b`

实现：

1. [_build_gradient_constraint_system](/home/kyrie/FaithC/src/faithc_infra/services/uv/method2_pipeline.py:234)
2. [_solve_poisson_uv](/home/kyrie/FaithC/src/faithc_infra/services/uv/method2_pipeline.py:420)

求解器后端：

1. 首选 CUDA PCG（可用时）
2. 自动回退 CPU SciPy CG

### 3.7 自适应锚点策略

当前锚点不是固定死数：

1. 可按连通分量大小自适应锚点数量（`min~max`）
2. 锚点分数融合：对应置信度 + 边界 + 曲率代理
3. 锚点为软约束，且权重按置信度缩放
4. 锚点目标 UV 优先来自采样对应统计，其次回退 nearest

实现：

1. [_component_minimal_anchor_ids](/home/kyrie/FaithC/src/faithc_infra/services/uv/method2_pipeline.py:301)
2. [_vertex_curvature_proxy](/home/kyrie/FaithC/src/faithc_infra/services/uv/method2_pipeline.py:251)

### 3.8 全局平移后对齐（post-align）

为降低整体 UV 漂移导致的颜色误差：

1. 先计算预测 UV 与目标 UV 的残差中位数
2. 施加全局平移（可配置最大位移）
3. 重算颜色重投影误差

实现：

1. [method2_pipeline.py](/home/kyrie/FaithC/src/faithc_infra/services/uv/method2_pipeline.py:1054)

## 4. 失败保护与回退

Method2 当前有完整保护链：

1. 无有效对应样本 -> nearest fallback
2. 无有效梯度约束 -> barycentric fallback
3. 线性求解异常 -> barycentric fallback

对应字段：

1. `uv_mode_used`
2. `uv_project_error`
3. `uv_m2_constraint_relaxation_used`
4. `uv_solve_num_constraints`

## 5. 关键配置（建议先调这些）

位于 `uv.method2`：

1. `min_samples_per_face`
2. `outlier_sigma` / `outlier_quantile`
3. `adaptive_anchor_enabled`
4. `anchor_min_points_per_component` / `anchor_max_points_per_component`
5. `adaptive_smooth_enabled` / `adaptive_smooth_beta`
6. `post_align_translation` / `post_align_max_shift`

公共求解位于 `uv.solve`：

1. `backend`（`auto/cuda_pcg/cpu_scipy`）
2. `lambda_smooth`
3. `pcg_max_iter` / `pcg_tol`
4. `anchor_weight`
5. `ridge_eps`

## 6. 主要诊断字段

优先关注：

1. `uv_solver_backend_used`
2. `uv_solve_num_constraints`
3. `uv_m2_jacobian_valid_ratio`
4. `uv_m2_anchor_count_total`
5. `uv_m2_adaptive_smooth_alpha_mean`
6. `uv_color_reproj_l1` / `uv_color_reproj_l2`
7. `uv_bad_tri_ratio` / `uv_flip_ratio`

