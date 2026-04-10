# Repository Guidelines

## Project Scope
This repository now contains two active layers:
- `src/faithcontour/`: core Faithful Contouring encoder/decoder library.
- `src/faithc_infra/`: experiment infrastructure, UV projection pipelines (Method2/Method4), reporting, preview launcher, and profiler integration.

Do not assume this is a `demo.py`-only repo anymore.

## Project Structure
- `src/faithcontour/`
  - Core FCT encode/decode pipeline (`encoder.py`, `decoder.py`, `qef_solver.py`, `segment_ops.py`).
- `src/faithc_infra/`
  - `cli.py`: `faithc-exp` entry (`run/eval/render/preview/report-stage2`).
  - `services/reconstruction.py`: reconstruction orchestration.
  - `services/uv_projector.py`: UV mode dispatch and output packing.
  - `services/uv/`: UV algorithms and support modules.
    - `method2_pipeline.py`
    - `method4_pipeline.py`
    - `correspondence.py`, `linear_solver.py`, `sampling.py`, `quality.py`, `options.py`
  - `profiler.py`: built-in runtime profiler.
- `tools/preview/run_faithc_preview.py`
  - Python bridge invoked by previewer / scripts.
- `viewer/opengl_previewer/`
  - C++ OpenGL + ImGui interactive previewer.
- `experiments/`
  - Configs, run outputs, reports, scripts.
- `docs/uv/`
  - Method docs (`README.md`, `method2_implementation.md`, `method4_implementation.md`).

## Build, Run, and Validation Commands
- Install editable package:
  - `pip install -e . --no-build-isolation`
- Core demo sanity:
  - `python demo.py -r 128`
- Experiment infra run:
  - `python -m faithc_infra.cli run -c experiments/configs/<config>.yaml`
- Launch previewer via CLI:
  - `python -m faithc_infra.cli preview --mesh assets/examples/pirateship.glb`
- Build C++ previewer:
  - `cmake -S viewer/opengl_previewer -B viewer/opengl_previewer/build -DCMAKE_BUILD_TYPE=Release`
  - `cmake --build viewer/opengl_previewer/build -j`
- Bridge direct run (useful for UV debugging):
  - `python tools/preview/run_faithc_preview.py --input <high.glb> --output <low.glb> --status <status.json> --resolution 128 --device cuda --project-uv --uv-mode method2`

## Profiler Defaults and Switches
- `faithc-exp run`: profiler default is ON.
- `faithc-exp preview`: profiler default is OFF.
- `tools/preview/run_faithc_preview.py`: profiler default is OFF.

Preferred switch (new):
- `--profiler` / `--no-profiler`

Legacy compatibility switch (still supported):
- `--profile` / `--no-profile`
- `--profile-top-k`
- `--profile-no-cprofile`

When adjusting profiler behavior, keep CLI and bridge behavior aligned.

## Coding Style
- Python: 4-space indentation, type hints, dataclasses where already used.
- C++ (previewer): keep style consistent with surrounding files; avoid broad style-only churn.
- Naming:
  - functions/modules: `snake_case`
  - classes: `PascalCase`
  - constants: `UPPER_SNAKE_CASE`
- Keep diffs minimal and targeted; do not mix unrelated refactors with algorithm changes.

## UV Pipeline Notes (Current Focus)
- Maintained UV methods:
  - `method2_gradient_poisson`
  - `method4_jacobian_injective`
- Any change in Method2/Method4 should be validated with:
  1. a bridge run producing status JSON,
  2. profiler sidecar review when performance-sensitive,
  3. key UV metrics (`uv_color_reproj_l1/l2`, `uv_bad_tri_ratio`, `uv_flip_ratio`, solver backend fields).

## Testing Guidelines
There is no full automated test suite yet. For each substantial change:
- run `python -m compileall` on touched modules,
- run at least one realistic mesh command (prefer preview bridge or `faithc-exp run` sample),
- confirm output artifacts exist (`status.json`, optional `.perf.json/.txt`, output mesh).

If adding tests, place them in `tests/test_<module>.py` with `pytest` conventions.

## Commit and PR Guidelines
- Use short imperative commit subjects (<= 72 chars).
- PR description should include:
  - purpose and scope,
  - environment (CUDA/PyTorch),
  - exact commands run for validation,
  - before/after metrics or screenshots for UV/preview changes.

## Operational Constraints
- This repo is CUDA-first; CPU fallback exists for some paths but is not the main target.
- Do not silently change UV default method routing or preview output JSON schema without documenting it in `docs/uv/` and PR notes.
