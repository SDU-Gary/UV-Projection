from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ReconstructionArtifact:
    sample_name: str
    high_mesh_normalized_path: Path
    low_mesh_path: Path
    tokens_path: Optional[Path]
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UVArtifact:
    sample_name: str
    low_mesh_uv_path: Path
    uv_map_path: Path
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsArtifact:
    sample_name: str
    metrics_path: Path
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderArtifact:
    sample_name: str
    render_dir: Path
    status: str
    command: List[str] = field(default_factory=list)
    log_path: Optional[Path] = None
    reason: Optional[str] = None


@dataclass
class SampleRecord:
    sample_name: str
    status: str
    paths: Dict[str, str]
    stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
