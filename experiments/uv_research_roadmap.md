# UV 映射后续实验路线图（离线、论文导向）

> 状态说明（2026-03）：当前代码库工程主线已收敛为 `method2_gradient_poisson` 与 `method4_jacobian_injective`。  
> 本文档中 Stage1/Stage3 相关内容保留为历史研究记录，不再作为默认实现目标。

## 1. 目标与边界

本路线图用于指导 FaithC 后续 UV 研究，固定前提如下：

1. 高模与低模共用同一张贴图（不走重烘焙）。
2. 任务是 UV 映射/传递，不是贴图重绘。
3. 以离线质量为核心，不考虑实时性。

总目标：

1. 在 hard 资产上显著超过当前线性基线。
2. 从工程可落地方法逐步过渡到理论完备度更高的方法。
3. 全部实验可通过现有 `faithc-exp` 工作流复现。


## 2. 统一实验协议（所有阶段共用）

### 2.1 冻结基线

对照方法固定为：

1. `uv.method = hybrid_global_opt`（当前线性主流程）。
2. 同一批对比实验中，重建参数保持一致，只改 UV 方法相关项。

### 2.2 数据分层

先使用仓库内模型，按难度分层：

1. Easy：`assets/examples/light_bulb.glb`，`assets/examples/cloth.glb`
2. Medium：`assets/examples/corgi_traveller.glb`，`assets/examples/pirateship.glb`
3. Hard：`assets/aksfbx.glb`，`assets/massive_nordic_coastal_cliff_vdssailfa_raw.glb`，`assets/rotarycannonfbx.glb`

扩展目标：Stage 1 稳定后，补齐不少于 10 个带 UV 贴图模型。

### 2.3 统一指标

几何/UV 合法性指标：

1. `uv_flip_ratio`
2. `uv_bad_tri_ratio`
3. `uv_stretch_p95`、`uv_stretch_p99`
4. `uv_out_of_bounds_ratio`

纹理一致性指标：

1. `uv_color_reproj_l1`
2. `uv_color_reproj_l2`

稳定性与成本指标：

1. success rate
2. 单样本总耗时
3. 失败类型统计

### 2.4 统一产物

沿用现有结构 `experiments/runs/<run_id>/`：

1. `run_meta.json`
2. `run_index.json`
3. `summary.csv`
4. 每样本的 `metrics.json`、`uv_stats.json`、导出 UV mesh

新增阶段分析报告路径：

1. `experiments/reports/stage1_*.md`
2. `experiments/reports/stage2_*.md`
3. `experiments/reports/stage3_*.md`


## 3. 分阶段计划

### Stage 1：方法1（最小重构，先修结构问题）

### 目标

1. 保留当前工程骨架，最小改动落地。
2. 修复专家高频指出的问题：分岛循环依赖、早期错误放大、seam 决策不稳定。
3. 快速拿到“明显优于线性基线”的结果。

### 方法定义

方法1采用“迭代岛感知 refinement”：

1. 初始对应 + 初始面岛标签。
2. 交替迭代（建议 3-5 轮）：对应更新 -> 岛标签更新 -> seam 更新 -> 全局求解。
3. 必须记录能量单调性与早停条件。

### 计划方法ID

1. `uv.method = method1_iterative_harmonic`

### 实验矩阵

1. 迭代轮数：`2, 3, 5`
2. 岛约束模式：`soft, strict`
3. seam 策略：`legacy, halfedge_island`

### 阶段通过门槛

在 medium + hard 子集上，相对基线满足：

1. `uv_bad_tri_ratio` 降低 >= 20%
2. `uv_color_reproj_l1` 降低 >= 20%
3. success rate 不低于基线


### Stage 2：方法2 与 方法4（新优化范式并行验证）

### 目标

验证两种新思路的上限与稳定性，选择后续论文主线候选。

### 方法2定义

梯度场传递 + 泊松重建：

1. 在高模提取 UV 梯度/Jacobian 场。
2. 将目标场稳健传递到低模。
3. 解泊松系统得到低模 UV（最小锚点约束）。

计划方法ID：

1. `uv.method = method2_gradient_poisson`

### 方法4定义

Jacobian 拟合 + 可逆非线性参数化：

1. 从对应集合估计每面 Jacobian 目标。
2. 通过注入式畸变能量（如 Symmetric Dirichlet）做非线性优化。
3. 使用 Newton 或准牛顿求解（复用既有非线性求解经验）。

计划方法ID：

1. `uv.method = method4_jacobian_injective`

### 实验矩阵

1. 方法2 vs 方法4 vs Stage1 最优版本
2. hard 资产优先
3. 方法4进行畸变权重扫描

### 阶段通过门槛

方法2或方法4至少一条满足：

1. 相对 Stage1 最优版本再提升 >= 10%（`uv_bad_tri_ratio` 与 `uv_color_reproj_l1`）
2. hard 资产上 `uv_flip_ratio` 接近 0


### Stage 3：方法5 与 OTM-UV（原型可行性验证）

### 目标

不追求首次即 SOTA，先验证高理论框架能跑通完整流程。

### 方法5原型

拓扑剪裁 + 度量/Jacobian 拟合 + 受约束全局优化：

1. 落地最小闭环实现。
2. 优先验证正确性和收敛行为，不做性能微调。

计划方法ID：

1. `uv.method = method5_metric_topology`

### OTM-UV原型

最优传输 + 度量传输基础模块：

1. 耦合构建模块
2. 度量传输模块
3. 约束 UV 求解模块

计划方法ID：

1. `uv.method = otm_uv_proto`

### 阶段通过门槛

1. 两个方法都能在至少 2 个 hard 模型上输出有效结果。
2. 全流程可复现（指标和日志完整）。
3. 形成清晰失败类型图谱与下一步工程化优先级。


## 4. 实施与实验组织

### 4.1 计划配置文件（实现阶段新增）

1. `experiments/configs/uv_stage1_method1.yaml`
2. `experiments/configs/uv_stage2_method2.yaml`
3. `experiments/configs/uv_stage2_method4.yaml`
4. `experiments/configs/uv_stage3_method5.yaml`
5. `experiments/configs/uv_stage3_otm_uv.yaml`

### 4.2 标准命令

运行批实验：

```bash
faithc-exp run -c experiments/configs/<config>.yaml
```

重算评估：

```bash
faithc-exp eval -r <run_id>
```

重跑渲染：

```bash
faithc-exp render -r <run_id>
```

交互预览：

```bash
faithc-exp preview --mesh <path_to_mesh>
```

### 4.3 可复现规范

1. 每个配置固定随机种子。
2. 同组对比实验固定重建参数。
3. 消融实验一次只改变一个关键因素。


## 5. 每阶段交付物（论文素材导向）

每阶段必须交付：

1. 主对比表（baseline vs 新方法）
2. 关键消融表（仅保留核心超参）
3. 可视化对比图（source/high、baseline low、新方法 low）
4. 失败案例页
5. 阶段结论（继续/调整/停止）

Stage 3 结束后必须沉淀：

1. 论文主方法候选
2. 强基线候选
3. 草稿核心章节材料（方法定义与优化目标、收敛性/合法性讨论、实验协议与主结果）


## 6. 时间盒与决策门

建议时间盒：

1. Stage 1：2 周
2. Stage 2：3 周
3. Stage 3：4 周

阶段门规则：

1. Stage N 进入 Stage N+1 前，必须在报告中明确写出是否达到门槛。
2. 若未达标，允许一次最多 1 周的定向修复迭代，再做 go/no-go 决策。
