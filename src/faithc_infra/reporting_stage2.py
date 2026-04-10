from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


LOWER_BETTER_METRICS = (
    "uv_bad_tri_ratio",
    "uv_color_reproj_l1",
    "uv_color_reproj_l2",
    "uv_flip_ratio",
)


@dataclass
class RunSummary:
    run_id: str
    config_path: str
    total_samples: int
    success_samples: int
    success_rate: float
    sample_rows: Dict[str, Dict[str, Any]]
    metric_means: Dict[str, Optional[float]]
    hard_flip_mean: Optional[float]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_value(stats: Mapping[str, Any], key: str) -> Optional[float]:
    raw = stats.get(key, None)
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def summarize_run(run_dir: Path, hard_sample_names: Sequence[str]) -> RunSummary:
    run_index = _load_json(run_dir / "run_index.json")
    samples = run_index.get("samples", [])
    total = len(samples)
    hard_set = {s.strip() for s in hard_sample_names if s.strip()}

    sample_rows: Dict[str, Dict[str, Any]] = {}
    metric_acc: Dict[str, List[float]] = {m: [] for m in LOWER_BETTER_METRICS}
    hard_flip_values: List[float] = []
    success = 0

    for sample in samples:
        name = str(sample.get("sample_name", "unknown"))
        status = str(sample.get("status", "failed"))
        stats = sample.get("stats", {}) or {}
        row = {
            "status": status,
            "uv_bad_tri_ratio": _metric_value(stats, "uv_bad_tri_ratio"),
            "uv_color_reproj_l1": _metric_value(stats, "uv_color_reproj_l1"),
            "uv_color_reproj_l2": _metric_value(stats, "uv_color_reproj_l2"),
            "uv_flip_ratio": _metric_value(stats, "uv_flip_ratio"),
            "uv_solver_stage": stats.get("uv_solver_stage"),
            "uv_m4_refine_status": stats.get("uv_m4_refine_status"),
        }
        sample_rows[name] = row

        if status != "success":
            continue
        success += 1

        for key in LOWER_BETTER_METRICS:
            val = row.get(key)
            if val is not None:
                metric_acc[key].append(float(val))

        if name in hard_set and row["uv_flip_ratio"] is not None:
            hard_flip_values.append(float(row["uv_flip_ratio"]))

    metric_means = {k: (mean(v) if len(v) > 0 else None) for k, v in metric_acc.items()}
    hard_flip_mean = mean(hard_flip_values) if len(hard_flip_values) > 0 else None
    success_rate = float(success / total) if total > 0 else 0.0

    return RunSummary(
        run_id=str(run_index.get("run_id", run_dir.name)),
        config_path=str(run_index.get("config_path", "")),
        total_samples=total,
        success_samples=success,
        success_rate=success_rate,
        sample_rows=sample_rows,
        metric_means=metric_means,
        hard_flip_mean=hard_flip_mean,
    )


def _relative_improvement(base: Optional[float], candidate: Optional[float]) -> Optional[float]:
    if base is None or candidate is None:
        return None
    denom = max(abs(base), 1e-12)
    return float((base - candidate) / denom)


def evaluate_stage2_gate(
    *,
    baseline: RunSummary,
    candidate: RunSummary,
    improve_threshold: float,
    hard_flip_threshold: float,
) -> Dict[str, Any]:
    bad_imp = _relative_improvement(
        baseline.metric_means.get("uv_bad_tri_ratio"),
        candidate.metric_means.get("uv_bad_tri_ratio"),
    )
    l1_imp = _relative_improvement(
        baseline.metric_means.get("uv_color_reproj_l1"),
        candidate.metric_means.get("uv_color_reproj_l1"),
    )

    pass_success = candidate.success_rate + 1e-12 >= baseline.success_rate
    pass_bad = (bad_imp is not None) and (bad_imp >= improve_threshold)
    pass_l1 = (l1_imp is not None) and (l1_imp >= improve_threshold)
    pass_flip = (candidate.hard_flip_mean is not None) and (candidate.hard_flip_mean <= hard_flip_threshold)

    return {
        "candidate_run_id": candidate.run_id,
        "baseline_run_id": baseline.run_id,
        "improve_threshold": improve_threshold,
        "hard_flip_threshold": hard_flip_threshold,
        "candidate_success_rate": candidate.success_rate,
        "baseline_success_rate": baseline.success_rate,
        "uv_bad_tri_ratio_improvement": bad_imp,
        "uv_color_reproj_l1_improvement": l1_imp,
        "candidate_hard_flip_mean": candidate.hard_flip_mean,
        "checks": {
            "success_rate_not_worse": pass_success,
            "bad_tri_improvement": pass_bad,
            "color_l1_improvement": pass_l1,
            "hard_flip_near_zero": pass_flip,
        },
        "passed": bool(pass_success and pass_bad and pass_l1 and pass_flip),
    }


def build_stage2_report(
    *,
    runs_dir: Path,
    baseline_run_id: str,
    method2_run_id: str,
    method4_run_id: str,
    hard_samples: Sequence[str],
    improve_threshold: float,
    hard_flip_threshold: float,
) -> Dict[str, Any]:
    baseline = summarize_run(runs_dir / baseline_run_id, hard_samples)
    m2 = summarize_run(runs_dir / method2_run_id, hard_samples)
    m4 = summarize_run(runs_dir / method4_run_id, hard_samples)

    gate_m2 = evaluate_stage2_gate(
        baseline=baseline,
        candidate=m2,
        improve_threshold=improve_threshold,
        hard_flip_threshold=hard_flip_threshold,
    )
    gate_m4 = evaluate_stage2_gate(
        baseline=baseline,
        candidate=m4,
        improve_threshold=improve_threshold,
        hard_flip_threshold=hard_flip_threshold,
    )

    sample_names = sorted(
        set(baseline.sample_rows.keys()) | set(m2.sample_rows.keys()) | set(m4.sample_rows.keys())
    )
    per_sample: List[Dict[str, Any]] = []
    for name in sample_names:
        row_b = baseline.sample_rows.get(name, {})
        row_m2 = m2.sample_rows.get(name, {})
        row_m4 = m4.sample_rows.get(name, {})
        per_sample.append(
            {
                "sample_name": name,
                "baseline_bad_tri": row_b.get("uv_bad_tri_ratio"),
                "method2_bad_tri": row_m2.get("uv_bad_tri_ratio"),
                "method4_bad_tri": row_m4.get("uv_bad_tri_ratio"),
                "baseline_l1": row_b.get("uv_color_reproj_l1"),
                "method2_l1": row_m2.get("uv_color_reproj_l1"),
                "method4_l1": row_m4.get("uv_color_reproj_l1"),
                "baseline_flip": row_b.get("uv_flip_ratio"),
                "method2_flip": row_m2.get("uv_flip_ratio"),
                "method4_flip": row_m4.get("uv_flip_ratio"),
                "baseline_status": row_b.get("status"),
                "method2_status": row_m2.get("status"),
                "method4_status": row_m4.get("status"),
                "method4_stage": row_m4.get("uv_solver_stage"),
                "method4_refine_status": row_m4.get("uv_m4_refine_status"),
            }
        )

    return {
        "hard_samples": list(hard_samples),
        "baseline": baseline.__dict__,
        "method2": m2.__dict__,
        "method4": m4.__dict__,
        "gate_method2": gate_m2,
        "gate_method4": gate_m4,
        "stage2_any_passed": bool(gate_m2["passed"] or gate_m4["passed"]),
        "per_sample": per_sample,
    }


def _fmt_num(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.{digits}f}%"


def _summary_table_rows(report: Dict[str, Any]) -> Iterable[str]:
    b = report["baseline"]
    m2 = report["method2"]
    m4 = report["method4"]
    rows = [
        "| Run | Success Rate | bad-tri(均值) | color L1(均值) | flip(均值) | Hard flip(均值) |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| baseline `{b['run_id']}` | {_fmt_pct(b['success_rate'])} | "
            f"{_fmt_num(b['metric_means']['uv_bad_tri_ratio'])} | "
            f"{_fmt_num(b['metric_means']['uv_color_reproj_l1'])} | "
            f"{_fmt_num(b['metric_means']['uv_flip_ratio'])} | "
            f"{_fmt_num(b['hard_flip_mean'])} |"
        ),
        (
            f"| method2 `{m2['run_id']}` | {_fmt_pct(m2['success_rate'])} | "
            f"{_fmt_num(m2['metric_means']['uv_bad_tri_ratio'])} | "
            f"{_fmt_num(m2['metric_means']['uv_color_reproj_l1'])} | "
            f"{_fmt_num(m2['metric_means']['uv_flip_ratio'])} | "
            f"{_fmt_num(m2['hard_flip_mean'])} |"
        ),
        (
            f"| method4 `{m4['run_id']}` | {_fmt_pct(m4['success_rate'])} | "
            f"{_fmt_num(m4['metric_means']['uv_bad_tri_ratio'])} | "
            f"{_fmt_num(m4['metric_means']['uv_color_reproj_l1'])} | "
            f"{_fmt_num(m4['metric_means']['uv_flip_ratio'])} | "
            f"{_fmt_num(m4['hard_flip_mean'])} |"
        ),
    ]
    return rows


def _gate_line(gate: Dict[str, Any]) -> str:
    checks = gate["checks"]
    return (
        f"- `{gate['candidate_run_id']}`: "
        f"passed={gate['passed']} | "
        f"success_rate_not_worse={checks['success_rate_not_worse']} | "
        f"bad_tri_improve={_fmt_pct(gate['uv_bad_tri_ratio_improvement'])} ({checks['bad_tri_improvement']}) | "
        f"color_l1_improve={_fmt_pct(gate['uv_color_reproj_l1_improvement'])} ({checks['color_l1_improvement']}) | "
        f"hard_flip_mean={_fmt_num(gate['candidate_hard_flip_mean'])} ({checks['hard_flip_near_zero']})"
    )


def render_stage2_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Stage2 UV Gate Report")
    lines.append("")
    lines.append("## Summary")
    lines.extend(_summary_table_rows(report))
    lines.append("")
    lines.append("## Gate Checks")
    lines.append(_gate_line(report["gate_method2"]))
    lines.append(_gate_line(report["gate_method4"]))
    lines.append(f"- Any passed: `{report['stage2_any_passed']}`")
    lines.append("")
    lines.append("## Per-Sample Snapshot")
    lines.append(
        "| sample | baseline bad/l1/flip | method2 bad/l1/flip | method4 bad/l1/flip | m4 stage | m4 refine |"
    )
    lines.append("|---|---:|---:|---:|---|---|")
    for row in report["per_sample"]:
        b = f"{_fmt_num(row['baseline_bad_tri'])}/{_fmt_num(row['baseline_l1'])}/{_fmt_num(row['baseline_flip'])}"
        m2 = f"{_fmt_num(row['method2_bad_tri'])}/{_fmt_num(row['method2_l1'])}/{_fmt_num(row['method2_flip'])}"
        m4 = f"{_fmt_num(row['method4_bad_tri'])}/{_fmt_num(row['method4_l1'])}/{_fmt_num(row['method4_flip'])}"
        lines.append(
            f"| {row['sample_name']} | {b} | {m2} | {m4} | "
            f"{row.get('method4_stage') or '-'} | {row.get('method4_refine_status') or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)

