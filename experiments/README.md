# Experiments Workspace

- `configs/`: reusable YAML configs.
- `runs/`: generated run outputs (ignored by git).

Recommended starter configs:
1. `configs/homework_baseline.yaml`: homework batch baseline.
2. `configs/mitsuba_smoke.yaml`: single-sample end-to-end smoke (reconstruction + Mitsuba3 render).
3. `configs/uv_stage2_method2.yaml`: Method2 UV 主线配置。
4. `configs/uv_stage2_method4.yaml`: Method4 UV 主线配置。

Each run creates a single directory with:
- `run_meta.json`
- `run_index.json`
- `summary.csv`
- per-sample artifacts (`mesh_low`, `uv_map`, `metrics`, `manifest`, optional render outputs)

Useful commands:
- `python -m faithc_infra.cli run -c experiments/configs/uv_stage2_method2.yaml`
- `python -m faithc_infra.cli run -c experiments/configs/uv_stage2_method4.yaml`
- `python -m faithc_infra.cli report-stage2 --baseline-run <stage1_run_id> --method2-run <stage2_m2_run_id> --method4-run <stage2_m4_run_id>`

Method 实现文档：

1. `docs/uv/method2_implementation.md`
2. `docs/uv/method4_implementation.md`
3. `docs/uv/README.md`

The `report-stage2` command writes:
- `experiments/reports/stage2_<timestamp>.json`
- `experiments/reports/stage2_<timestamp>.md`

and evaluates Stage2 gate checks from `experiments/uv_research_roadmap.md`:
- success rate not worse than baseline
- `uv_bad_tri_ratio` relative improvement >= 10%
- `uv_color_reproj_l1` relative improvement >= 10%
- hard-sample mean `uv_flip_ratio` <= 0.05
