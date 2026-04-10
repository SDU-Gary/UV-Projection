# Mitsuba3 Rendering Adapter

This folder defines the Mitsuba3 rendering backend for experiment outputs.

## Environment Setup (Conda)
Recommended in `faithc` env:

```bash
conda activate faithc
pip install mitsuba drjit
```

## Runtime Contract
`RenderService` reads `manifest.json`, picks mesh in order:
1. `meshes.low_uv`
2. `meshes.low`
3. `meshes.high`

If needed, mesh is converted to OBJ for Mitsuba3.

## Presets
- Preset files live in `presets/*.yaml`.
- Default preset: `presets/default.yaml`.

## Optional Standalone Script
You can run a standalone test script:

```bash
FAITHC_MANIFEST=/abs/path/manifest.json \
FAITHC_OUTPUT_DIR=/abs/path/render \
FAITHC_PRESET_PATH=renderer/mitsuba3/presets/default.yaml \
FAITHC_VARIANT=cuda_ad_rgb \
FAITHC_SPP=64 \
python renderer/mitsuba3/scripts/render_scene.py
```
