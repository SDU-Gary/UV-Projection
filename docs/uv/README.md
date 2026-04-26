# UV Pipeline Notes (Method2 / Method4)

本目录用于记录 FaithC 当前维护中的 UV 主方法实现细节。

当前建议主路径：

1. `method2_gradient_poisson`
2. `method4_jacobian_injective`
3. `method25_projected_jacobian_injective`（实验性，不改默认）

低模来源说明：

1. 当前 UV / seam 主线默认仍可使用 `FaithC` 输出低模。
2. 为了排查“FaithC 输出拓扑/绕序不健康”对半边与 UV 岛提取的干扰，预览桥接与重建服务现已支持 `pymeshlab_qem` 作为对照低模后端。
   - 当前经验默认值是“允许真实 decimation”的组合：`preserve_boundary=false`、`preserve_normal=false`、`preserve_topology=false`。
   - 若把这些保守约束重新打开，PyMeshLab 可能只做顶点焊接而几乎不降面；状态 JSON 会通过 `reconstruction_pymeshlab_target_achieved=false` 暴露这一点。
   - 当前实现会在 decimation 后自动执行非流形修复，并最多重复若干轮顶点修复，直到蝴蝶结顶点清零或达到迭代上限。
3. 相关状态字段：
   - `reconstruction_backend_requested`
   - `reconstruction_backend_used`
   - `reconstruction_pymeshlab_*`
4. 当低模输入绕序不一致时，`halfedge_island` 会优先回退到 `halfedge_numpy_fallback`，并把原因写入：
   - `uv_seam_extraction_backend`
   - `uv_openmesh_fallback_reason`
5. CUDA BVH 运行时现在通过共享守卫统一初始化，避免 preview / diagnostics / reconstruction 三条路径行为不一致。
   - Method2 / Method4 / Hybrid / barycentric BVH 路径都会在公共入口先执行 Atom3d CUDA runtime ensure。
   - 相关状态字段：
     - `atom3d_runtime_patched`
     - `atom3d_runtime_reason`
     - `atom3d_arch`
     - `atom3d_cumtv_module`
     - `atom3d_bvh_module`
     - `atom3d_smoke_test`
     - `atom3d_bvh_smoke_test`

说明：

1. `method4` 依赖 `method2` 的内部状态作为初始化和回退基线。
2. `method25` 也是在 `method2` 内部状态之上工作，但会先用残差场 projector 重建一版面级目标 Jacobian，再做一次线性初始化，然后交给 `method4` 的非线性约束求解器。
3. `hybrid_global_opt`、`nearest_vertex`、`barycentric_closest_point` 仍保留在代码里用于回退或历史复现实验，但不作为当前主优化路线。

文档列表：

1. [Method2 实现文档](./method2_implementation.md)
2. [Method4 实现文档](./method4_implementation.md)
3. [Method2.5 实现文档](./method25_implementation.md)

推荐阅读顺序：

1. 先读 Method2（线性主解 + 诊断 + 回退策略）
2. 再读 Method4（非线性 refinement，基于 Method2 初始化）
3. 最后读 Method2.5（Exp13 projector + Method4 solver 的实验缝合）
