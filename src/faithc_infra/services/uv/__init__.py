from __future__ import annotations

from .options import DEFAULT_OPTIONS, METHOD_ALIASES, deep_merge_dict, resolve_seam_validation_settings

__all__ = [
    "DEFAULT_OPTIONS",
    "METHOD_ALIASES",
    "compute_cached_high_face_uv_islands",
    "deep_merge_dict",
    "resolve_seam_validation_settings",
    "run_halfedge_island_pipeline",
    "sample_low_mesh",
]


def __getattr__(name: str):
    if name in {"compute_cached_high_face_uv_islands", "run_halfedge_island_pipeline"}:
        from .island_pipeline import compute_cached_high_face_uv_islands, run_halfedge_island_pipeline

        mapping = {
            "compute_cached_high_face_uv_islands": compute_cached_high_face_uv_islands,
            "run_halfedge_island_pipeline": run_halfedge_island_pipeline,
        }
        return mapping[name]
    if name == "sample_low_mesh":
        from .sampling import sample_low_mesh

        return sample_low_mesh
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
