# UV Pipeline Notes (Method2 / Method4)

本目录用于记录 FaithC 当前维护中的 UV 主方法实现细节。

当前建议主路径：

1. `method2_gradient_poisson`
2. `method4_jacobian_injective`

说明：

1. `method4` 依赖 `method2` 的内部状态作为初始化和回退基线。
2. `hybrid_global_opt`、`nearest_vertex`、`barycentric_closest_point` 仍保留在代码里用于回退或历史复现实验，但不作为当前主优化路线。

文档列表：

1. [Method2 实现文档](./method2_implementation.md)
2. [Method4 实现文档](./method4_implementation.md)

推荐阅读顺序：

1. 先读 Method2（线性主解 + 诊断 + 回退策略）
2. 再读 Method4（非线性 refinement，基于 Method2 初始化）

