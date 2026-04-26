# Method2: Gradient-Transferred Poisson UV Mapping

## 1. 文档目标

本文档面向当前代码库中的 `method2_gradient_poisson` 实际实现，给出一份接近国际会议论文方法节风格的、但严格服从源码行为的说明。它不是理想化算法草图，而是对以下主线的实现级复原：

- 入口：`src/faithc_infra/services/uv_projector.py`
- 主流程：`src/faithc_infra/services/uv/method2_pipeline.py`
- 采样与对应：`src/faithc_infra/services/uv/sampling.py`、`src/faithc_infra/services/uv/correspondence.py`
- 岛语义与切缝：`src/faithc_infra/services/uv/island_pipeline.py`、`src/faithc_infra/services/uv/semantic_transfer.py`
- 稀疏求解：`src/faithc_infra/services/uv/linear_solver.py`

默认实验配置见 `experiments/configs/uv_stage2_method2.yaml`。当前主线并不是“单纯从对应点直接拟合 UV”，而是一个“高模局部 UV 微分场转移 + 岛约束 + 泊松重建”的 pipeline。

## 2. 问题定义与目标

给定：

- 高模三角网格 `M_h = (V_h, F_h)`，其每个顶点有已知 UV `U_h`
- 由高模减面得到的低模三角网格 `M_l = (V_l, F_l)`
- 可选的高模底色纹理图像 `I`

目标是在低模上求得顶点 UV `U_l`，使得：

1. 低模能够与高模共用同一张纹理贴图；
2. 低模在纹理细节、局部方向场、接缝结构上尽量继承高模；
3. 在低模几何显著简化、甚至高低模不完全逐顶点对应时，仍保持求解稳定。

Method2 的核心思想不是直接回归 `U_l`，而是先从高模估计局部 UV Jacobian 场，再将该 Jacobian 场投影到低模面上，最后通过稀疏泊松系统恢复低模顶点 UV。

## 3. 方法总览

当前实现的真实主线可写成：

1. 在高模上建立 CUDA BVH、法向与 UV 访问上下文。
2. 在高模上计算 UV island；在低模上建立与这些 island 对应的语义面分区。
3. 若启用 `halfedge_island`，沿低模语义切缝复制顶点，构造求解网格 `solve_mesh`。
4. 在低模面内做面积自适应采样。
5. 对每个采样点在高模上做双向法线射线匹配，失败时退化为最近点 UDF 投影。
6. 将匹配到的高模面 UV Jacobian 以稳健统计方式聚合到低模面。
7. 由面 Jacobian 构造边差分约束，结合平滑项、软锚点项和 ridge 项建立线性系统。
8. 按 UV island 分块或整体求解两个标量场 `u, v`。
9. 用采样残差的中位数做一次全局平移后对齐，减少整体漂移。
10. 输出低模 UV、质量指标、求解诊断和可导出的 split topology。

如果把该实现写成优化问题，其主目标可近似表述为

\[
\min_{U_l}
\|A U_l - b\|_2^2
 \lambda \, \mathcal{R}(U_l)
 \mathcal{A}(U_l)
 \varepsilon \|U_l\|_2^2,
\]

其中：

- `A U_l = b` 对应“低模边差分应逼近目标 Jacobian 诱导的 UV 差分”
- `\mathcal{R}` 是基于拉普拉斯的平滑项
- `\mathcal{A}` 是基于高模对应统计构造的软锚点项
- `\varepsilon` 是数值稳定的 ridge 正则

但真实实现不是先写统一能量再交给黑盒求解器，而是按工程阶段显式构建每一项。

## 4. 高模 UV 岛分析与低模语义切缝

### 4.1 高模 UV island 的定义

高模 island 的计算在 `compute_high_face_uv_islands` 中完成。其规则不是直接读现成 UV chart，而是按以下条件在高模面之间建图：

1. 先将高模顶点按三维位置做 `position_eps` 级别的 weld；
2. 对每条焊接后的无向边，收集其两侧三角面；
3. 若两侧三角面在该公共边两端的 UV 坐标都在 `uv_eps` 范围内一致，则认为这两面属于同一 UV 连通域；
4. 否则该边被视为 UV seam；
5. 对面邻接图做连通分量分析，得到高模 face-level island label。

因此，Method2 中的 island 是一个严格的“面级 UV 连通域”概念，而不是几何连通域。

### 4.2 默认主线：`halfedge_island`

当前 stage2 主线配置使用：

- `uv.seam.strategy = halfedge_island`
- `uv.seam.transfer_sampling_mode = four_point_soft_flood`

这意味着 Method2 默认不是旧的“legacy seam heuristic”，而是先显式构造低模语义切缝，再在 split 后的拓扑上解 UV。

### 4.3 低模预清理

在 `run_halfedge_island_pipeline` 中，低模首先进入 `sanitize_mesh_for_halfedge`：

1. 用 PyMeshLab 修复 non-manifold edge；
2. 修复 non-manifold vertex；
3. 尝试重定向面朝向；
4. 删除退化面；
5. 用 Trimesh 重新修复法向与多连通体朝向一致性。

这样做的目的不是改善几何质量本身，而是保证后续 OpenMesh seam 提取、半边切缝和语义分区验证稳定。

### 4.4 高模 island 语义向低模面的传递

在默认 `four_point_soft_flood` 路径中，每个低模面不是只投一个质心，而是投四个采样点：

\[
\{(1/3,1/3,1/3),\ (0.6,0.2,0.2),\ (0.2,0.6,0.2),\ (0.2,0.2,0.6)\}.
\]

对于每个采样点：

1. 从点沿局部法线和反法线各发一条射线；
2. 射线起点做极小偏移，减小 `t \approx 0` 自相交；
3. 使用 `transfer_max_dist_ratio * bbox_diag` 作为最大投影距离；
4. 若命中距离很近，则放宽法向夹角检查；
5. 否则要求法向夹角小于阈值；
6. 选取正反两个方向中更近且有效的命中；
7. 记录命中的高模 face island label，而不是高模 face id。

随后，对每个低模面收集四个岛标签样本，建立离散概率分布：

- `top1 label / prob`
- `top2 label / prob`
- 熵
- 候选标签数

再执行一个“软优先级泛洪 + ICM”过程：

1. 选择高置信、与次优标签分隔明显的强种子面；
2. 在低模面邻接图上做带代价的优先级传播；
3. 对未决面用邻域投票回填；
4. 用 ICM 做若干轮局部 relabel；
5. 计算最终标签置信度，并将低置信或未知面标为 `conflict`。

这个阶段的核心作用不是直接给出 UV，而是给 Method2 一个可靠的低模“面属于哪个高模 UV island”的先验。

### 4.5 语义后处理

`island_pipeline.py` 在软传播之后还做两类清理：

1. 吸收过小的语义连通块；
2. 吸收微小非主壳层组件到主壳层。

其目标是减少碎片化 label，避免后续 seam graph 出现大量微小开链和无意义切割。

### 4.6 Seam 提取、验证与顶点复制

有了低模面标签后，系统在低模上提取 seam edge：

1. 若 OpenMesh 可用且输入朝向一致，则优先用 OpenMesh 提取；
2. 否则回退到 numpy halfedge 逻辑；
3. 相邻两面标签不同的边被视为 seam；
4. 可选地把边界边也计入 seam。

之后执行两类验证：

1. 拓扑验证：seam component 是否闭合，是否存在 branch vertex；
2. 分区验证：移除 seam 后，每个面连通块是否保持纯标签。

需要强调的是：这些验证结果在当前 Method2 中主要被记录到诊断字段里。只要 `halfedge_island` 流程成功产出高模 island 和低模语义分区，主求解通常仍继续执行；当前实现并不会因为 `validation_ok=false` 就统一终止整个 Method2。

最后，`split_vertices_along_cut_edges` 沿 seam 复制顶点。复制规则不是简单复制所有 seam 顶点，而是：

1. 对每个顶点收集其 incident faces；
2. 在这些 faces 之间，仅保留“未跨 seam 的邻接”；
3. 将 incident faces 划分成若干局部 connected components；
4. 每个 component 获得自己的顶点副本。

这一步生成真正的求解网格 `solve_mesh`。在默认主线中，泊松系统是在这个 split 后的拓扑上求解的。

## 5. 低模采样

Method2 对低模的采样是面内随机重心采样，不是顶点采样，也不是蓝噪声采样。

对每个低模面 `f`，采样数设置为

\[
n_f = \mathrm{clip}\left(
\mathrm{round}\left(\text{base\_per\_face}\cdot \sqrt{A_f / \bar{A}}\right),
\text{min\_per\_face},
\text{max\_per\_face}
\right),
\]

其中 `A_f` 是该面面积，`\bar A` 是全局平均面面积。

每个采样点记录：

- 三维坐标
- 重心坐标
- 所属低模面 id
- 面法线
- 面积权重 `A_f / n_f`

因此，小面不会完全失去采样，而大面也不会线性增长到不可控。

## 6. 高低模对应：双向射线 + 最近点回退

### 6.1 高模上下文

Method2 只在 CUDA 路径下工作。若 `resolve_device(device) != "cuda"`，整个方法直接回退到 barycentric 投影，不进入泊松链路。

在 CUDA 主线中，系统先构建：

- 高模顶点张量
- 高模面张量
- 高模 UV 张量
- 高模面法向张量
- 基于 `atom3d.MeshBVH` 的 BVH
- 高模包围盒对角线长度

### 6.2 主对应策略

对于每个低模采样点：

1. 分别沿法线和反法线发射一条射线；
2. 命中需满足：
   - 命中 face id 合法；
   - 命中距离有限；
   - 命中面法向与射线方向点积大于 `normal_dot_min`；
3. 若正反两个方向都有效，取距离更近者；
4. 用命中高模三角面的重心坐标插值得到目标 UV。

### 6.3 失败时的回退

若双向射线均失败，则转用 `bvh.udf(... return_closest=True, return_uvw=True, return_face_ids=True)`：

1. 找最近高模三角面；
2. 用最近点或返回的 `uvw` 做三角形内插；
3. 仅要求该面法向与采样法向点积非负；
4. 标记该采样为 `fallback_used_mask=True`。

### 6.4 重要实现事实

当前 Method2 并未把 island guard 作为 `correspond_points_hybrid` 的在线约束传入。源码虽然支持在对应阶段做 island-compatible filtering，但 Method2 主线没有启用这一接口。当前实际策略是：

1. 先做几何对应；
2. 再根据低模面 island 语义对采样进行后过滤。

这与“在对应阶段直接约束岛兼容性”在实现上不是一回事。

## 7. 岛一致性过滤与样本有效域

对应完成后，Method2 不会立即进入 Jacobian 聚合，而是继续做多级过滤。

### 7.1 Legacy 路径下的低模面 island 估计

如果不是 `halfedge_island`，系统会用已有采样对应做一个低模面主岛标签统计：

1. 对每个低模面，收集其有效采样命中的高模 island；
2. 取出现次数最多的 island 作为该面标签；
3. 若一个面同时收到多个 island 的投票，则标记为 `conflict`；
4. 置信度定义为 dominant vote 占比。

### 7.2 跨缝面检测

在非 `halfedge_island` 路径下，系统会检查每个低模面的目标 UV 样本跨度：

\[
\mathrm{span}_f = \sqrt{(u_{\max}-u_{\min})^2 + (v_{\max}-v_{\min})^2}.
\]

若跨度超过 `uv_span_threshold`，则认为该面跨 seam；配置允许时，这些面的采样被排除出求解。

### 7.3 可选 island guard

`method2.use_island_guard=true` 时，Method2 会进行一轮基于低模面语义的前置过滤：

1. 冲突面上的采样全部剔除；
2. 若 `allow_unknown=false`，未知面上的采样也剔除。

但这只是预过滤，不是最终约束。

### 7.4 强制严格岛一致性

只要高模 island 信息存在，Method2 无论 `use_island_guard` 是否开启，都会执行一轮更强的严格过滤：

1. 对每个采样，取其所属低模面的目标 island `\hat s_f`；
2. 取该采样实际命中的高模 face island `s(p)`；
3. 仅保留满足
   - `\hat s_f >= 0`
   - 该面不冲突
   - `s(p) = \hat s_f`
   的采样；
4. 同时将未知或冲突的低模面从 `face_active_mask` 中移除。

因此，实际主线中真正决定可参与泊松求解的样本，是“几何有效 + 高模 Jacobian 有效 + 岛一致”的交集。

这是 Method2 的一个关键实现点：即使 `use_island_guard=false`，只要前面成功建立了 island 语义，后面依然存在强制的 island-consistent filtering。

## 8. 高模面 UV Jacobian 的计算

Method2 的核心传递对象是高模每个三角面的 `2 x 3` UV Jacobian。

对高模面 `t = (p_0,p_1,p_2)`，先定义三维边向量：

\[
e_1 = p_1 - p_0,\quad e_2 = p_2 - p_0.
\]

再定义 UV 边向量：

\[
\Delta u_1 = U_1 - U_0,\quad \Delta u_2 = U_2 - U_0.
\]

源码先计算三维局部几何的伪逆

\[
P_t^{+} \in \mathbb{R}^{2\times 3},
\]

然后构造

\[
J_t^h =
\begin{bmatrix}
\Delta u_1 & \Delta u_2
\end{bmatrix}
P_t^{+}
\in \mathbb{R}^{2\times 3}.
\]

这里 `J_t^h` 的含义是：给定该三角面上的三维位移向量 `d \in \mathbb{R}^3`，预测 UV 位移为 `J_t^h d`。

只有当面局部 Gram 矩阵行列式大于 `1e-18` 时，该 Jacobian 才被标记为有效。

## 9. 从高模采样 Jacobian 到低模面 Jacobian

### 9.1 每个低模面的目标 Jacobian 来源

对每个保留下来的采样点 `p_i`：

- 已知其所属低模面 `f_i`
- 已知其命中的高模面 `g_i`
- 已知高模面 Jacobian `J_{g_i}^h`
- 已知采样权重 `w_i`

Method2 通过所有 `f_i = f` 的样本，把高模 Jacobian 聚合成低模面的目标 Jacobian `J_f^*`。

### 9.2 采样权重

Method2 的每个采样权重由三项乘积组成：

\[
w_i = w_i^{area}\, w_i^{corr}\, w_i^{tex},
\]

其中：

1. `w_i^{area}`：来自面面积均分，即 `A_f / n_f`
2. `w_i^{corr}`：若采样来自主射线命中则为 `1`，若来自 UDF fallback 则为 `fallback_weight`
3. `w_i^{tex}`：若有贴图，则按目标 UV 处纹理梯度幅值放大，高纹理梯度区域被赋予更高权重

于是，Method2 会更信任：

- 大面贡献
- 非 fallback 的稳定对应
- 纹理梯度高的位置

### 9.3 稳健聚合

对同一低模面内的 Jacobian 样本集合，Method2 采用如下稳健聚合流程：

1. 对样本 Jacobian 做带权均值初始化；
2. 进行若干轮 IRLS；
3. IRLS 的残差范数基于 Jacobian Frobenius 距离；
4. Huber 权重通过 `huber_delta` 控制；
5. 完成 IRLS 后再做显式异常值剔除：
   - `median + sigma * MAD`
   - `quantile(q)`
   - 两者取较大阈值
6. 若剔除后保留样本不足 `min_samples_per_face`，则回退为保留最近的若干样本；
7. 对最终聚合结果执行一次 `scale resuscitation`：
   - 保持聚合 Jacobian 的方向
   - 把其范数恢复到样本 Jacobian 的加权平均范数

最后得到：

- `face_jac[f] = J_f^*`
- `face_weights[f] = \sum_i w_i`
- `face_cov_trace[f]`：聚合样本的离散度指标
- `face_valid[f]`：该面是否拥有足够稳定的 Jacobian

### 9.4 小样本快速路径

当一个低模面的有效样本数不超过 `perf_fast_small_group_threshold` 时，Method2 可以跳过完整 IRLS 和 outlier rejection，直接走快速均值路径，以减少小组统计开销。

### 9.5 约束放松

若第一次聚合后没有任何有效面 Jacobian，Method2 会自动做一次放松：

- `min_samples_per_face = 1`
- `outlier_sigma = 0`
- `outlier_quantile = 1`
- `irls_iters = 1`

这不是新的算法分支，而是“尽量保住至少一个可解约束集”的保护机制。

## 10. 自适应平滑权重

Method2 并不是对所有面施加相同的平滑强度。它首先把 `face_cov_trace` 归一化：

\[
\tilde c_f = c_f / \mathrm{median}(c),
\]

然后定义面平滑系数

\[
\alpha_f = \mathrm{clip}\left(
\frac{1}{1 + \beta \max(\tilde c_f,0)},
\alpha_{\min},
\alpha_{\max}
\right).
\]

解释如下：

- 若一个低模面接收到的高模 Jacobian 样本很一致，则 `c_f` 小，`alpha_f` 大；
- 若一个低模面样本分歧很强，则 `c_f` 大，`alpha_f` 小。

这些 `\alpha_f` 被投影到边上，构造成加权拉普拉斯。于是 Method2 会在局部 Jacobian 场稳定处更强平滑，在不稳定处减弱平滑，避免把明显多模态的 Jacobian 混得过于模糊。

## 11. 软锚点的构造

Method2 不采用硬 Dirichlet elimination，而是始终使用“强软锚点”。

### 11.1 顶点目标 UV 的统计估计

在构建锚点之前，Method2 先从采样对应恢复每个低模顶点的候选目标 UV：

1. 对每个有效采样，把其目标 UV 按该采样在所属低模三角形上的重心权重分配给三个顶点；
2. 对每个顶点累积：
   - UV 加权和
   - 权重和
   - 置信度和
3. 若某顶点累积权重大于零，则其锚点目标 UV 为加权平均；
4. 否则为 `NaN`，后续由 nearest-vertex UV 补齐。

### 11.2 顶点置信度

每个顶点的采样支持度被归一化到 `[0,1]`，形成 `anchor_vertex_confidence`。归一化尺度使用 95 分位数而不是最大值，以减轻极端大样本顶点的影响。

### 11.3 锚点选择

锚点选择在连通分量上进行。默认模式为 `component_minimal`：

1. 对每个连通分量确定锚点数量 `k`；
2. 若启用自适应锚点，则

\[
k = \mathrm{clip}\left(
\left\lceil \frac{|V_{comp}|}{\text{target\_vertices\_per\_anchor}} \right\rceil,
k_{\min},
k_{\max}
\right);
\]

3. 顶点分数来自
   - 对应置信度
   - 边界 boost
   - 曲率代理 boost
4. 第一个锚点取最高分顶点；
5. 后续锚点用 farthest-point-style 选择，同时乘上分数项，兼顾空间分散与可靠性。

曲率代理并不是离散主曲率，而是基于顶点法向在邻边上的变化量：

\[
\kappa(v) \approx 1 - \langle n_i, n_j \rangle.
\]

### 11.4 一个重要实现细节

`anchor_mode = none` 在当前实现中并不表示“完全无锚点”。它仍然会为每个连通分量选择至少一个最小锚点，以防止系统的平移零空间漂移。因此，Method2 实际上始终保留分量级锚定。

### 11.5 软锚点写入系统

若锚点集合为 `\mathcal{S}`，则对每个锚点顶点 `v`，Method2 在系统矩阵对角线上增加

\[
w_v = \text{anchor\_weight} \cdot \tilde c_v,
\]

并同步在右端项中加入 `w_v \hat U_v`。

其中 `\tilde c_v` 是经过 `anchor_confidence_floor` 抬底后的置信度。也就是说，置信度会调节锚点强度，但不会把锚点削弱到完全没有数值作用。

## 12. 泊松系统的构造

### 12.1 面 Jacobian 到边差分约束

对每个有效低模面 `f=(i,j,k)`，Method2 不直接约束三角形参数化矩阵，而是选择三条有向边：

- `(i,j)`
- `(i,k)`
- `(j,k)`

记对应三维边向量为 `e`，则目标 UV 差分为

\[
\delta_f(e) = J_f^* e \in \mathbb{R}^2.
\]

于是每条边给出两条标量约束：

\[
u_b - u_a \approx \delta_f(e)_u,\quad
v_b - v_a \approx \delta_f(e)_v.
\]

每个面的三条边约束都乘以 `sqrt(face_weight[f])`，得到统一的稀疏矩阵 `A` 以及两组右端项 `rhs_u, rhs_v`。

### 12.2 平滑项

Method2 的线性系统主矩阵首先是

\[
M = A^\top A.
\]

若 `lambda_smooth > 0`，再加上平滑项：

- 若启用 adaptive smooth，则使用基于 `face_alpha` 的 edge-weighted Laplacian；
- 否则使用 `uniform` 或 `cotan` Laplacian。

因此

\[
M \leftarrow A^\top A + \lambda L.
\]

### 12.3 Ridge 项

为避免病态系统，Method2 无条件再加

\[
\varepsilon I,\quad \varepsilon = \text{ridge\_eps}.
\]

### 12.4 最终系统

对 `u`、`v` 两个通道分别求解：

\[
M u = A^\top rhs_u + b_{anchor,u},
\quad
M v = A^\top rhs_v + b_{anchor,v}.
\]

也就是说，Method2 实际求解的是两个共享系统矩阵、不同右端项的 SPD 线性系统。

## 13. 分岛求解与整体求解

### 13.1 为什么按 island 分解

若前面成功建立了低模面的 island 标签，并且 `solve_per_island=true`，Method2 会按岛把系统拆成多个子问题：

1. 对每个 island 收集其涉及的低模面；
2. 从全局 `A` 中抽取只属于该岛的约束行；
3. 提取该岛所覆盖的局部子网格；
4. 在局部子网格上独立求解 UV；
5. 把局部解写回全局顶点数组。

### 13.2 作用

分岛求解的主要好处是：

1. 消除不同 UV island 间的无意义耦合；
2. 降低系统规模；
3. 让每个 chart 的漂移与锚定更加局部化。

在默认 `halfedge_island` 主线上，由于 seam 已导致顶点复制，岛间顶点重叠通常应接近零。源码仍然保留 `overlap_vertices` 统计，以便监控未完全 split 的异常情况。

## 14. 线性求解器后端

Method2 支持两层后端：

1. 首选 CUDA sparse PCG；
2. 失败时回退 CPU SciPy 族求解器。

### 14.1 CUDA PCG

若请求 `auto` 且设备是 CUDA，系统会：

1. 把 CSR 系统转为 torch sparse COO；
2. 取对角元作为 Jacobi 预条件；
3. 以 `rhs / diag` 作为初值；
4. 用自写 PCG 分别求 `u` 和 `v`。

如果：

- 初始残差非有限
- 分母退化
- PCG 未在设定迭代数内收敛

则抛异常并切换 CPU 路径。

### 14.2 CPU 路径

CPU 稳健求解器顺序大致为：

1. `cg`
2. `cg + jacobi`
3. `cholmod`，若环境可用
4. `pyamg + cg`，若环境可用
5. `spsolve`
6. `lsmr`

因此 Method2 的数值策略不是单一 solver，而是一个“尽量保解”的后备链。

## 15. 全局平移后对齐

泊松场积分后的 UV 往往具有整体平移自由度。虽然软锚点已经抑制了漂移，但源码仍然额外做一轮全局平移校正。

步骤为：

1. 用已解出的顶点 UV 在采样点上做重心插值，得到预测 UV；
2. 计算采样级残差

\[
r_i = U_i^{target} - U_i^{pred};
\]

3. 取残差的二维中位数 `median(r_i)` 作为候选平移；
4. 若样本数不少于 `post_align_min_samples`，则把整个 UV 场平移该向量；
5. 若平移模长过大，则裁剪到 `post_align_max_shift`。

这个步骤只做平移，不做旋转、缩放或仿射重配准。它的目标是修正“整体偏置”，不是改变局部形变。

## 16. 质量评估与输出

Method2 输出的不只是 `mapped_uv`，还包括：

- `uv_color_reproj_l1 / l2`
- `uv_flip_ratio`
- `uv_bad_tri_ratio`
- `uv_stretch_p95 / p99`
- 求解器残差、迭代和后端信息
- 岛语义、切缝和 split 统计
- Jacobian 聚合统计

若使用了 `halfedge_island` 并发生顶点复制，则：

1. `quality_mesh` 使用 split 后拓扑；
2. 导出 mesh 时也优先导出 split topology；
3. 保证 UV 顶点数与导出网格顶点数一致。

## 17. 失败保护与回退链

Method2 不是“一旦有一步失败就崩溃”的实现，而是一条分层保护链：

1. 无 CUDA：直接回退 barycentric
2. `halfedge_island` 若未能成功产出可用的高模 island 结果，则在非严格路径下回退 barycentric；若处于严格路径，异常会上抛
3. 无有效对应样本：回退 nearest
4. 无有效 Jacobian 约束：回退 barycentric
5. 求解器异常：回退 barycentric

因此，Method2 在工程上被实现成：

- 主线尽量用 Jacobian-Poisson
- 一旦关键条件不足，退回更朴素但稳定的映射方式

## 18. 与默认实验配置的对应关系

`experiments/configs/uv_stage2_method2.yaml` 对应的实际主线可以概括为：

1. `seam.strategy = halfedge_island`
2. `transfer_sampling_mode = four_point_soft_flood`
3. `solve_per_island = true`
4. `adaptive_anchor_enabled = true`
5. `adaptive_smooth_enabled = true`
6. `post_align_translation = true`
7. `laplacian_mode = cotan`
8. `backend = auto`

因此，当前 Method2 的“论文级真实主方法”并不是旧的单图全局 smooth Poisson，而是：

“基于高模 UV island 语义转移、低模 chart-aware split、稳健 Jacobian 聚合与分岛泊松重建的 UV 映射方法”。

## 19. 重要实现细节与容易误解之处

### 19.1 `use_island_guard=false` 不等于没有 island 过滤

即使这个开关关闭，只要高模 island 和低模面岛标签已经建立，Method2 仍然会执行强制的 strict island consistency filter。

### 19.2 `anchor_mode=none` 不是无锚点

它仍然保留每个连通分量至少一个软锚点，只是减少锚点数。

### 19.3 `hard_anchor_*` 当前不参与真实求解

源码中显式写明：当前实现只用 strong soft anchors，不做 hard Dirichlet elimination。相关配置项保留主要是兼容性原因。

### 19.4 `local_vertex_split` 在 Method2 主线中不是实际控制开关

在当前 Method2 里，`halfedge_island` 路径由 `split_vertices_along_cut_edges` 直接决定是否复制顶点；legacy 路径则没有像 hybrid 那样单独应用 face-local vertex split。

### 19.5 seam validation 当前更接近诊断而非统一 hard gate

`uv_seam_validation_ok`、`uv_island_validation_ok` 等字段会被完整记录，但当前 Method2 并不会在所有 validation failure 上统一回退。真正决定是否中断或回退的，是更上游的关键产物是否缺失，例如高模 island 根本未成功建立。

### 19.6 `correspondence.normal_weight` 与 `fallback_k` 当前并未进入 Method2 主计算图

这两个配置项仍保留在配置和 preview CLI 中，但当前 `correspond_points_hybrid` 主逻辑实际使用的是：

- `normal_dot_min`
- `ray_max_dist_ratio`
- `fallback_weight`
- `bvh_chunk_size`

因此，阅读或写论文时不应把未被调用的旧参数误写成 Method2 的核心组成。

## 20. 可作为论文方法节的精炼表述

如果把当前实现压缩成一段论文方法描述，可写为：

> 我们首先在高模上基于 UV 连续性构建 face-level UV island 分区，并通过四点投影与软优先级传播将其转移到低模面上，得到与高模 chart 语义一致的低模分区。随后沿分区边界在低模上执行拓扑切缝，将原始低模转化为可独立参数化的多个 chart。对于每个低模面，我们在面内进行面积自适应采样，并通过双向法线射线加最近点回退在高模上建立稳健对应。高模已有 UV 被转化为面级 `2x3` Jacobian 场，并按低模面上的采样集合以 IRLS、Huber 和分位数/MAD 异常抑制进行稳健聚合，从而得到低模目标 Jacobian 场。我们进一步依据 Jacobian 聚合方差自适应调节平滑强度，并利用采样级目标 UV 统计构造置信度加权的软锚点。最终，方法通过求解带有拉普拉斯正则、软锚点和 ridge 项的稀疏泊松系统，在每个 UV island 内独立恢复低模顶点 UV，并在解后施加一次基于采样残差中位数的全局平移校正。该实现能够在保持共享纹理坐标系一致性的同时，把高模局部纹理方向场和接缝结构稳定迁移到减面后的低模上。 

## 21. 结论

从实现上看，Method2 的本质不是单一“解泊松系统”四个字，而是一个由以下关键模块共同构成的系统：

1. 高模 UV 岛解析
2. 低模 chart 语义转移
3. seam-aware 拓扑切分
4. 稳健高低模对应
5. Jacobian 场转移与聚合
6. 自适应平滑
7. 置信度软锚点
8. 分岛泊松重建
9. 后平移校正

其中真正决定该方法效果的，不只是最后那一步线性求解，而是前面“如何把高模局部 UV 微分结构变成低模上可信、可分块、可积分的 Jacobian 约束场”。
