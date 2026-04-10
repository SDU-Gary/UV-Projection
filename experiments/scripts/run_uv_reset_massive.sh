#!/usr/bin/env bash
set -euo pipefail

# One-click UV reset experiment runner:
# - Asset: massive_nordic_coastal_cliff_vdssailfa_raw.glb
# - Resolutions: 64 and 128 (default)
# - Pipelines: Stage1 legacy baseline, Stage2 Method2, Stage2 Method4
# - Auto-generate stage2 gate reports

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ASSET_PATH="${ASSET_PATH:-assets/massive_nordic_coastal_cliff_vdssailfa_raw.glb}"
SAMPLE_NAME="${SAMPLE_NAME:-massive_cliff}"
RESOLUTIONS_RAW="${RESOLUTIONS:-64 128}"
WORK_DIR="${WORK_DIR:-/tmp/faithc_uv_reset_$(date +%Y%m%d-%H%M%S)}"

mkdir -p "$WORK_DIR"

if [[ ! -f "$ASSET_PATH" ]]; then
  echo "ERROR: asset not found: $ASSET_PATH"
  exit 1
fi

run_ids_csv="$WORK_DIR/run_ids.csv"
{
  echo "resolution,baseline_run,method2_run,method4_run,report_json,report_md"
} > "$run_ids_csv"

create_cfg_from_template() {
  local template_path="$1"
  local output_path="$2"
  local resolution="$3"

  "$PYTHON_BIN" - "$template_path" "$output_path" "$ASSET_PATH" "$SAMPLE_NAME" "$resolution" <<'PY'
import sys
import yaml

template_path, output_path, asset_path, sample_name, resolution = sys.argv[1:]
resolution = int(resolution)

with open(template_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["data"]["samples"] = [{"name": sample_name, "high_mesh": asset_path}]

recon = cfg["pipeline"]["reconstruction"]
recon["resolution"] = resolution
if resolution >= 128:
    recon["retry_resolutions"] = [max(128, resolution * 2)]
else:
    recon["retry_resolutions"] = [128, 256]

uv_method = str(cfg["pipeline"]["uv"]["method"])
method_tag = {
    "hybrid_global_opt": "s1legacy",
    "method2_gradient_poisson": "s2m2",
    "method4_jacobian_injective": "s2m4",
}.get(uv_method, "uv")
cfg["project"]["name"] = f"faithc-{method_tag}-massive-r{resolution}"
cfg["project"]["run_prefix"] = f"{method_tag}-massive-r{resolution}"

with open(output_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

run_pipeline() {
  local label="$1"
  local cfg_path="$2"
  local out
  local status
  local run_id

  echo
  echo "===== Running: $label ====="
  echo "Config: $cfg_path"

  set +e
  out="$("$PYTHON_BIN" -m faithc_infra.cli run -c "$cfg_path" 2>&1)"
  status=$?
  set -e

  echo "$out"
  if [[ $status -ne 0 ]]; then
    echo "ERROR: run failed for $label"
    return "$status"
  fi

  run_id="$(printf '%s\n' "$out" | sed -n 's/^run_id=//p' | tail -n 1)"
  if [[ -z "$run_id" ]]; then
    echo "ERROR: cannot parse run_id for $label"
    return 2
  fi
  printf '%s\n' "$run_id"
}

run_report() {
  local baseline_run_id="$1"
  local method2_run_id="$2"
  local method4_run_id="$3"
  local out
  local report_json
  local report_md

  echo
  echo "===== Running: stage2 report ====="
  echo "baseline=$baseline_run_id method2=$method2_run_id method4=$method4_run_id"

  out="$("$PYTHON_BIN" -m faithc_infra.cli report-stage2 \
    --baseline-run "$baseline_run_id" \
    --method2-run "$method2_run_id" \
    --method4-run "$method4_run_id" \
    --hard-samples "$SAMPLE_NAME" 2>&1)"
  echo "$out"

  report_json="$(printf '%s\n' "$out" | sed -n 's/^stage2_report_json=//p' | tail -n 1)"
  report_md="$(printf '%s\n' "$out" | sed -n 's/^stage2_report_md=//p' | tail -n 1)"
  if [[ -z "$report_json" || -z "$report_md" ]]; then
    echo "ERROR: cannot parse report output paths"
    return 2
  fi

  printf '%s,%s\n' "$report_json" "$report_md"
}

read -r -a RESOLUTIONS_ARR <<< "$RESOLUTIONS_RAW"

for res in "${RESOLUTIONS_ARR[@]}"; do
  echo
  echo "###############################"
  echo "### Resolution: $res"
  echo "###############################"

  cfg_baseline="$WORK_DIR/uv_stage1_legacy_r${res}.yaml"
  cfg_m2="$WORK_DIR/uv_stage2_method2_r${res}.yaml"
  cfg_m4="$WORK_DIR/uv_stage2_method4_r${res}.yaml"

  create_cfg_from_template "experiments/configs/uv_stage1_legacy_once.yaml" "$cfg_baseline" "$res"
  create_cfg_from_template "experiments/configs/uv_stage2_method2.yaml" "$cfg_m2" "$res"
  create_cfg_from_template "experiments/configs/uv_stage2_method4.yaml" "$cfg_m4" "$res"

  baseline_run_id="$(run_pipeline "baseline-r${res}" "$cfg_baseline" | tail -n 1)"
  method2_run_id="$(run_pipeline "method2-r${res}" "$cfg_m2" | tail -n 1)"
  method4_run_id="$(run_pipeline "method4-r${res}" "$cfg_m4" | tail -n 1)"

  report_paths="$(run_report "$baseline_run_id" "$method2_run_id" "$method4_run_id" | tail -n 1)"
  report_json="${report_paths%%,*}"
  report_md="${report_paths##*,}"

  echo "${res},${baseline_run_id},${method2_run_id},${method4_run_id},${report_json},${report_md}" >> "$run_ids_csv"
done

echo
echo "Done. Summary CSV:"
echo "  $run_ids_csv"
echo
cat "$run_ids_csv"

