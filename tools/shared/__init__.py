"""
Shared utility functions for UV-Projection tools.

This module provides common functionality used across multiple diagnostic
and analysis scripts, reducing code duplication and ensuring consistency.
"""

import sys
import json
from pathlib import Path
from typing import Any, Dict

# Repository paths
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

# Ensure src is in Python path
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_mesh(mesh_path: Path, process: bool = True):
    """
    Load a mesh from file using trimesh.

    Args:
        mesh_path: Path to mesh file
        process: Whether to process the mesh (remove duplicates, etc.)

    Returns:
        trimesh.Trimesh: Loaded mesh object

    Raises:
        FileNotFoundError: If mesh file doesn't exist
        ValueError: If mesh file is invalid
    """
    import trimesh

    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    try:
        mesh = trimesh.load(mesh_path, force="mesh", process=process)
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            raise ValueError(f"Invalid mesh file: {mesh_path}")
        return mesh
    except Exception as e:
        raise ValueError(f"Failed to load mesh {mesh_path}: {e}")


def sanitize_json(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable types.

    Handles numpy types, Path objects, and other non-serializable types.

    Args:
        obj: Object to sanitize

    Returns:
        JSON-serializable version of the object
    """
    import numpy as np

    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    return obj


def save_json(data: Dict[str, Any], output_path: Path, indent: int = 2) -> None:
    """
    Save data to JSON file with proper sanitization.

    Args:
        data: Dictionary to save
        output_path: Path to output JSON file
        indent: JSON indentation level
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(sanitize_json(data), f, indent=indent)


def load_json(input_path: Path) -> Dict[str, Any]:
    """
    Load data from JSON file.

    Args:
        input_path: Path to input JSON file

    Returns:
        Dictionary with loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is invalid JSON
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"JSON file not found: {input_path}")

    with input_path.open("r") as f:
        return json.load(f)


def setup_tool_environment():
    """
    Setup common environment for tool scripts.

    This function ensures proper Python path configuration
    and can be extended for other common setup tasks.
    """
    # Already handled at module import
    pass


def print_tool_header(tool_name: str, description: str = ""):
    """
    Print a formatted header for tool output.

    Args:
        tool_name: Name of the tool
        description: Optional description
    """
    print("=" * 60)
    print(f"  {tool_name}")
    if description:
        print(f"  {description}")
    print("=" * 60)


__all__ = [
    "REPO_ROOT",
    "SRC_ROOT",
    "load_mesh",
    "sanitize_json",
    "save_json",
    "load_json",
    "setup_tool_environment",
    "print_tool_header",
]
