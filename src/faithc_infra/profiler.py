from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return x


@dataclass
class ProfilerConfig:
    enabled: bool = True
    cprofile_enabled: bool = True
    tracemalloc_enabled: bool = True
    top_k: int = 50
    sync_cuda_on_stop: bool = True


class ExecutionProfiler:
    """Generic process profiler (timing + hotspots + memory + CUDA diagnostics)."""

    def __init__(
        self,
        *,
        name: str,
        config: Optional[ProfilerConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = str(name)
        self.config = config or ProfilerConfig()
        self.metadata = dict(metadata or {})

        self._started = False
        self._stopped = False
        self._start_wall = 0.0
        self._start_cpu = 0.0
        self._end_wall = 0.0
        self._end_cpu = 0.0
        self._started_at_iso: Optional[str] = None
        self._ended_at_iso: Optional[str] = None

        self._prof: Optional[cProfile.Profile] = None
        self._cprofile_text_by_cum: str = ""
        self._cprofile_text_by_self: str = ""

        self._tracemalloc_started = False
        self._stage_events: List[Dict[str, Any]] = []
        self._report: Optional[Dict[str, Any]] = None

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._started_at_iso = _now_iso()
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        if not self.config.enabled:
            return

        if self.config.tracemalloc_enabled:
            try:
                tracemalloc.start()
                self._tracemalloc_started = True
            except Exception:
                self._tracemalloc_started = False

        if self.config.cprofile_enabled:
            self._prof = cProfile.Profile()
            self._prof.enable()

    def stop(self, *, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._stopped and self._report is not None:
            if extra:
                self._report.setdefault("extra", {}).update(extra)
            return self._report

        if not self._started:
            self.start()

        if self._prof is not None:
            try:
                self._prof.disable()
            except Exception:
                pass

        self._end_wall = time.perf_counter()
        self._end_cpu = time.process_time()
        self._ended_at_iso = _now_iso()

        memory = self._collect_memory_stats()
        cuda = self._collect_cuda_stats(sync=self.config.sync_cuda_on_stop)
        hotspots = self._collect_hotspots()
        stage_summary = self._summarize_stage_events()

        report: Dict[str, Any] = {
            "name": self.name,
            "enabled": bool(self.config.enabled),
            "started_at": self._started_at_iso,
            "ended_at": self._ended_at_iso,
            "wall_time_seconds": _safe_float(self._end_wall - self._start_wall),
            "cpu_time_seconds": _safe_float(self._end_cpu - self._start_cpu),
            "metadata": self.metadata,
            "memory": memory,
            "cuda": cuda,
            "stage_events": self._stage_events,
            "stage_summary": stage_summary,
            "hotspots": hotspots,
            "extra": dict(extra or {}),
        }
        self._report = report
        self._stopped = True
        return report

    @contextmanager
    def step(self, name: str, **meta: Any) -> Iterator[None]:
        if not self.config.enabled:
            yield
            return
        if not self._started:
            self.start()
        t0 = time.perf_counter()
        status = "ok"
        err: Optional[str] = None
        try:
            yield
        except Exception as exc:
            status = "error"
            err = str(exc)
            raise
        finally:
            dt = time.perf_counter() - t0
            event = {
                "name": str(name),
                "seconds": _safe_float(dt),
                "status": status,
            }
            if meta:
                event["meta"] = meta
            if err is not None:
                event["error"] = err
            self._stage_events.append(event)

    def write_reports(
        self,
        *,
        json_path: Path,
        text_path: Optional[Path] = None,
        report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = report if report is not None else self.stop()

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)

        if text_path is not None:
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text = self._render_text_report(data)
            text_path.write_text(text, encoding="utf-8")
        return data

    def _collect_memory_stats(self) -> Dict[str, Any]:
        mem: Dict[str, Any] = {}
        if self._tracemalloc_started:
            try:
                cur, peak = tracemalloc.get_traced_memory()
                mem["tracemalloc_current_mb"] = _safe_float(cur / (1024.0 * 1024.0))
                mem["tracemalloc_peak_mb"] = _safe_float(peak / (1024.0 * 1024.0))
            except Exception:
                pass
            try:
                tracemalloc.stop()
            except Exception:
                pass
            self._tracemalloc_started = False

        try:
            import resource

            ru = resource.getrusage(resource.RUSAGE_SELF)
            # Linux ru_maxrss is KiB.
            mem["max_rss_mb"] = _safe_float(float(ru.ru_maxrss) / 1024.0)
            mem["user_cpu_seconds"] = _safe_float(ru.ru_utime)
            mem["sys_cpu_seconds"] = _safe_float(ru.ru_stime)
        except Exception:
            pass
        return mem

    def _collect_cuda_stats(self, *, sync: bool) -> Dict[str, Any]:
        out: Dict[str, Any] = {"available": False, "devices": []}
        try:
            import torch
            import warnings
        except Exception:
            return out

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cuda_available = bool(torch.cuda.is_available())
        except Exception:
            return out
        if not cuda_available:
            return out

        out["available"] = True
        n = int(torch.cuda.device_count())
        devices: List[Dict[str, Any]] = []
        for i in range(n):
            dev = torch.device(f"cuda:{i}")
            if sync:
                try:
                    torch.cuda.synchronize(dev)
                except Exception:
                    pass
            d: Dict[str, Any] = {"index": i}
            try:
                props = torch.cuda.get_device_properties(dev)
                d["name"] = props.name
                d["capability"] = f"{props.major}.{props.minor}"
                d["total_memory_mb"] = _safe_float(float(props.total_memory) / (1024.0 * 1024.0))
            except Exception:
                pass
            try:
                d["memory_allocated_mb"] = _safe_float(
                    float(torch.cuda.memory_allocated(dev)) / (1024.0 * 1024.0)
                )
                d["memory_reserved_mb"] = _safe_float(
                    float(torch.cuda.memory_reserved(dev)) / (1024.0 * 1024.0)
                )
                d["max_memory_allocated_mb"] = _safe_float(
                    float(torch.cuda.max_memory_allocated(dev)) / (1024.0 * 1024.0)
                )
                d["max_memory_reserved_mb"] = _safe_float(
                    float(torch.cuda.max_memory_reserved(dev)) / (1024.0 * 1024.0)
                )
            except Exception:
                pass
            devices.append(d)
        out["devices"] = devices
        return out

    def _collect_hotspots(self) -> Dict[str, Any]:
        if self._prof is None:
            return {"by_cumulative": [], "by_self": []}

        stats = pstats.Stats(self._prof).strip_dirs()
        rows: List[Dict[str, Any]] = []
        for (filename, lineno, funcname), (cc, nc, tt, ct, _callers) in stats.stats.items():
            rows.append(
                {
                    "file": str(filename),
                    "line": int(lineno),
                    "function": str(funcname),
                    "primitive_calls": int(cc),
                    "total_calls": int(nc),
                    "self_seconds": _safe_float(tt),
                    "cum_seconds": _safe_float(ct),
                }
            )

        by_cum = sorted(rows, key=lambda x: float(x.get("cum_seconds") or 0.0), reverse=True)[: self.config.top_k]
        by_self = sorted(rows, key=lambda x: float(x.get("self_seconds") or 0.0), reverse=True)[: self.config.top_k]

        sio_c = io.StringIO()
        sio_s = io.StringIO()
        try:
            pstats.Stats(self._prof, stream=sio_c).strip_dirs().sort_stats("cumtime").print_stats(self.config.top_k)
            pstats.Stats(self._prof, stream=sio_s).strip_dirs().sort_stats("tottime").print_stats(self.config.top_k)
            self._cprofile_text_by_cum = sio_c.getvalue()
            self._cprofile_text_by_self = sio_s.getvalue()
        except Exception:
            self._cprofile_text_by_cum = ""
            self._cprofile_text_by_self = ""

        return {"by_cumulative": by_cum, "by_self": by_self}

    def _summarize_stage_events(self) -> List[Dict[str, Any]]:
        acc: Dict[str, Dict[str, Any]] = {}
        for ev in self._stage_events:
            name = str(ev.get("name", "unknown"))
            sec = float(ev.get("seconds") or 0.0)
            cur = acc.get(name)
            if cur is None:
                cur = {
                    "name": name,
                    "count": 0,
                    "total_seconds": 0.0,
                    "avg_seconds": 0.0,
                    "max_seconds": 0.0,
                    "error_count": 0,
                }
                acc[name] = cur
            cur["count"] += 1
            cur["total_seconds"] += sec
            cur["max_seconds"] = max(float(cur["max_seconds"]), sec)
            if str(ev.get("status", "ok")) != "ok":
                cur["error_count"] += 1

        out: List[Dict[str, Any]] = []
        for v in acc.values():
            c = max(1, int(v["count"]))
            v["avg_seconds"] = _safe_float(float(v["total_seconds"]) / c)
            v["total_seconds"] = _safe_float(v["total_seconds"])
            v["max_seconds"] = _safe_float(v["max_seconds"])
            out.append(v)
        out.sort(key=lambda x: float(x.get("total_seconds") or 0.0), reverse=True)
        return out

    def _render_text_report(self, report: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append(f"Profiler: {report.get('name')}")
        lines.append(f"Started: {report.get('started_at')}")
        lines.append(f"Ended:   {report.get('ended_at')}")
        lines.append(f"Wall(s): {report.get('wall_time_seconds')}")
        lines.append(f"CPU(s):  {report.get('cpu_time_seconds')}")
        lines.append("")

        mem = report.get("memory", {})
        if mem:
            lines.append("Memory:")
            for k, v in mem.items():
                lines.append(f"  - {k}: {v}")
            lines.append("")

        cuda = report.get("cuda", {})
        lines.append("CUDA:")
        lines.append(f"  - available: {cuda.get('available')}")
        for d in cuda.get("devices", []):
            lines.append(
                "  - "
                + f"cuda:{d.get('index')} {d.get('name')} "
                + f"alloc={d.get('memory_allocated_mb')}MB "
                + f"peak_alloc={d.get('max_memory_allocated_mb')}MB"
            )
        lines.append("")

        lines.append("Top Stage Summary:")
        for row in report.get("stage_summary", [])[:30]:
            lines.append(
                "  - "
                + f"{row.get('name')}: total={row.get('total_seconds')}s "
                + f"count={row.get('count')} avg={row.get('avg_seconds')}s "
                + f"max={row.get('max_seconds')}s errors={row.get('error_count')}"
            )
        lines.append("")

        if self._cprofile_text_by_cum:
            lines.append("Hotspots by cumulative time:")
            lines.append(self._cprofile_text_by_cum.rstrip())
            lines.append("")
        if self._cprofile_text_by_self:
            lines.append("Hotspots by self time:")
            lines.append(self._cprofile_text_by_self.rstrip())
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"
