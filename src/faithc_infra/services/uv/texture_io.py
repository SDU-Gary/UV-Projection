from __future__ import annotations

import base64
import io
import json
import struct
import urllib.parse
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import trimesh
from PIL import Image


def extract_uv(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        return None
    uv = np.asarray(uv)
    if uv.ndim != 2 or uv.shape[1] != 2:
        return None
    if uv.shape[0] != len(mesh.vertices):
        return None
    return uv.astype(np.float32)


def extract_basecolor_image(mesh: trimesh.Trimesh):
    material = getattr(mesh.visual, "material", None)
    if material is None:
        return None
    return getattr(material, "image", None)


def get_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    vn = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if vn.shape[0] != len(mesh.vertices):
        vn = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
        vn[:, 1] = 1.0
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(n, 1e-8)


def resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def decode_data_uri(uri: str) -> Optional[bytes]:
    if not uri.startswith("data:"):
        return None
    comma = uri.find(",")
    if comma < 0:
        return None
    header = uri[:comma]
    payload = uri[comma + 1 :]
    if ";base64" in header:
        return base64.b64decode(payload)
    return urllib.parse.unquote_to_bytes(payload)


def load_image_from_bytes(image_bytes: bytes):
    if not image_bytes:
        return None
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")
    return image


def extract_basecolor_image_from_gltf_path(path: Path):
    path = path.resolve()
    ext = path.suffix.lower()

    gltf = None
    base_dir = path.parent
    bin_chunk = None

    if ext == ".glb":
        blob = path.read_bytes()
        if len(blob) < 20:
            return None

        magic, version, _ = struct.unpack_from("<4sII", blob, 0)
        if magic != b"glTF" or version != 2:
            return None

        offset = 12
        json_chunk = None
        while offset + 8 <= len(blob):
            chunk_len, chunk_type = struct.unpack_from("<II", blob, offset)
            offset += 8
            chunk = blob[offset : offset + chunk_len]
            offset += chunk_len
            if chunk_type == 0x4E4F534A:
                json_chunk = chunk
            elif chunk_type == 0x004E4942:
                bin_chunk = chunk

        if json_chunk is None:
            return None
        gltf = json.loads(json_chunk.decode("utf-8"))
    elif ext == ".gltf":
        with path.open("r", encoding="utf-8") as f:
            gltf = json.load(f)
    else:
        return None

    if not isinstance(gltf, dict):
        return None

    images = gltf.get("images", [])
    textures = gltf.get("textures", [])
    materials = gltf.get("materials", [])
    buffer_views = gltf.get("bufferViews", [])
    buffers = gltf.get("buffers", [])

    if not images:
        return None

    image_index = None
    for material in materials:
        pbr = material.get("pbrMetallicRoughness", {})
        bc = pbr.get("baseColorTexture", {})
        tex_idx = bc.get("index")
        if isinstance(tex_idx, int) and 0 <= tex_idx < len(textures):
            source_idx = textures[tex_idx].get("source")
            if isinstance(source_idx, int) and 0 <= source_idx < len(images):
                image_index = source_idx
                break

    if image_index is None:
        for tex in textures:
            source_idx = tex.get("source")
            if isinstance(source_idx, int) and 0 <= source_idx < len(images):
                image_index = source_idx
                break

    if image_index is None:
        image_index = 0

    image_entry = images[image_index]
    uri = image_entry.get("uri")
    if isinstance(uri, str):
        data = decode_data_uri(uri)
        if data is not None:
            return load_image_from_bytes(data)
        file_path = base_dir / urllib.parse.unquote(uri)
        if file_path.exists():
            return load_image_from_bytes(file_path.read_bytes())
        return None

    view_idx = image_entry.get("bufferView")
    if not isinstance(view_idx, int) or view_idx < 0 or view_idx >= len(buffer_views):
        return None

    view = buffer_views[view_idx]
    buffer_idx = int(view.get("buffer", 0))
    view_offset = int(view.get("byteOffset", 0))
    view_length = int(view.get("byteLength", 0))
    if view_length <= 0:
        return None

    data_blob = None
    if ext == ".glb":
        if buffer_idx != 0 or bin_chunk is None:
            return None
        data_blob = bin_chunk
    else:
        if buffer_idx < 0 or buffer_idx >= len(buffers):
            return None
        buffer_uri = buffers[buffer_idx].get("uri")
        if not isinstance(buffer_uri, str):
            return None
        data_blob = decode_data_uri(buffer_uri)
        if data_blob is None:
            buffer_path = base_dir / urllib.parse.unquote(buffer_uri)
            if not buffer_path.exists():
                return None
            data_blob = buffer_path.read_bytes()

    if data_blob is None:
        return None
    if view_offset < 0 or view_offset + view_length > len(data_blob):
        return None

    image_bytes = data_blob[view_offset : view_offset + view_length]
    return load_image_from_bytes(image_bytes)


def resolve_basecolor_image(mesh: trimesh.Trimesh, texture_source_path: Optional[Path]) -> Tuple[Any, Optional[str]]:
    image = extract_basecolor_image(mesh)
    if image is not None:
        return image, "trimesh_material"

    if texture_source_path is not None:
        image = extract_basecolor_image_from_gltf_path(Path(texture_source_path))
        if image is not None:
            return image, "gltf_image_chunk"

    return None, None


__all__ = [
    "decode_data_uri",
    "extract_basecolor_image",
    "extract_basecolor_image_from_gltf_path",
    "extract_uv",
    "get_vertex_normals",
    "load_image_from_bytes",
    "resolve_basecolor_image",
    "resolve_device",
]

