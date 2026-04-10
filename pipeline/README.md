# FaithC Pipeline Notes

The experiment pipeline is driven by `faithc-exp`:

- `faithc-exp run -c <config.yaml>`: execute reconstruction/uv/eval/render.
- `faithc-exp eval -r <run_id>`: recompute metrics.
- `faithc-exp render -r <run_id>`: rerun Mitsuba3 rendering.
- `faithc-exp preview [--mesh <path>]`: launch OpenGL interactive previewer.

OpenGL previewer build steps:

```bash
cd viewer/opengl_previewer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Preview mode uses `tools/preview/run_faithc_preview.py` to call FaithC reconstruction asynchronously.
