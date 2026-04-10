from __future__ import annotations

from typing import Any, Dict

import numpy as np
import trimesh


def sample_low_mesh(low_mesh: trimesh.Trimesh, cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    faces = np.asarray(low_mesh.faces, dtype=np.int64)
    verts = np.asarray(low_mesh.vertices, dtype=np.float64)
    face_normals = np.asarray(low_mesh.face_normals, dtype=np.float64)
    face_areas = np.asarray(low_mesh.area_faces, dtype=np.float64)

    mean_area = float(np.mean(face_areas)) if len(face_areas) > 0 else 1.0
    mean_area = max(mean_area, 1e-12)
    area_scale = np.sqrt(np.maximum(face_areas / mean_area, 1e-12))

    base = int(cfg.get("base_per_face", 4))
    counts = np.rint(base * area_scale).astype(np.int32)
    counts = np.clip(counts, int(cfg.get("min_per_face", 3)), int(cfg.get("max_per_face", 12)))

    total = int(np.sum(counts))
    points = np.zeros((total, 3), dtype=np.float64)
    bary = np.zeros((total, 3), dtype=np.float64)
    normals = np.zeros((total, 3), dtype=np.float64)
    face_ids = np.zeros(total, dtype=np.int64)
    area_weights = np.zeros(total, dtype=np.float64)

    rng = np.random.default_rng(int(cfg.get("seed", 12345)))

    cursor = 0
    for fi in range(len(faces)):
        n = int(counts[fi])
        if n <= 0:
            continue

        r1 = rng.random(n)
        r2 = rng.random(n)
        s1 = np.sqrt(r1)
        w0 = 1.0 - s1
        w1 = s1 * (1.0 - r2)
        w2 = s1 * r2

        tri = verts[faces[fi]]
        pts = w0[:, None] * tri[0] + w1[:, None] * tri[1] + w2[:, None] * tri[2]

        points[cursor : cursor + n] = pts
        bary[cursor : cursor + n, 0] = w0
        bary[cursor : cursor + n, 1] = w1
        bary[cursor : cursor + n, 2] = w2
        normals[cursor : cursor + n] = face_normals[fi]
        face_ids[cursor : cursor + n] = fi
        area_weights[cursor : cursor + n] = max(1e-12, face_areas[fi] / n)
        cursor += n

    return {
        "points": points,
        "bary": bary,
        "normals": normals,
        "face_ids": face_ids,
        "area_weights": area_weights,
    }

