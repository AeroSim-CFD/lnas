"""Tests comparing LnasFormat.from_file (Python) output against stl2lnas (Rust) output.

Both tools should produce geometrically identical results for the same STL input:
same vertex set, same triangle connectivity with consistent winding, same surfaces.
"""

import pathlib
import subprocess
import tempfile

import numpy as np
import pytest

from lnas import LnasFormat

_FIXTURE_DIR = pathlib.Path("fixture")

_STL_FILES = [
    "cube.stl",
    "cube_no_norm.stl",
    "cylinder.stl",
]


def _run_stl2lnas(stl_path: pathlib.Path, output_path: pathlib.Path):
    result = subprocess.run(
        ["stl2lnas", "-f", str(stl_path), "-o", str(output_path), "--overwrite"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stl2lnas failed: {result.stderr}"


def _canonical_triangles(lnas: LnasFormat) -> np.ndarray:
    """Return a sorted canonical representation of triangles for geometry comparison.

    Each triangle (p0, p1, p2) is cyclically rotated so that its lexicographically
    smallest vertex comes first, preserving winding order.  All triangles are then
    sorted lexicographically, giving an order-independent fingerprint of the mesh.

    Shape of output: (N, 3, 3) — N triangles × 3 vertices × 3 coordinates.
    """
    tv = lnas.geometry.triangle_vertices.copy()  # (N, 3, 3)
    canonical = np.empty_like(tv)

    for i, tri in enumerate(tv):
        # Pick the cyclic rotation that starts at the lex-smallest vertex
        rotations = [np.roll(tri, -k, axis=0) for k in range(3)]
        canonical[i] = min(rotations, key=lambda t: t.tolist())

    # Sort triangles lexicographically using their flattened form
    order = np.lexsort(canonical.reshape(len(canonical), -1).T[::-1])
    return canonical[order]


@pytest.mark.parametrize("stl_name", _STL_FILES)
def test_from_stl_matches_stl2lnas(stl_name: str):
    """Python from_file and Rust stl2lnas must produce the same geometry."""
    stl_path = _FIXTURE_DIR / stl_name

    with tempfile.NamedTemporaryFile(suffix=".lnas", delete=False) as f:
        rust_lnas_path = pathlib.Path(f.name)

    _run_stl2lnas(stl_path, rust_lnas_path)

    py_lnas = LnasFormat.from_file(stl_path)
    rust_lnas = LnasFormat.from_file(rust_lnas_path)

    # Same triangle and vertex count
    assert len(py_lnas.geometry.triangles) == len(rust_lnas.geometry.triangles), (
        f"Triangle count mismatch: py={len(py_lnas.geometry.triangles)}, "
        f"rust={len(rust_lnas.geometry.triangles)}"
    )
    assert len(py_lnas.geometry.vertices) == len(rust_lnas.geometry.vertices), (
        f"Vertex count mismatch: py={len(py_lnas.geometry.vertices)}, "
        f"rust={len(rust_lnas.geometry.vertices)}"
    )

    # Same canonical triangle geometry (vertex positions + winding)
    py_canon = _canonical_triangles(py_lnas)
    rust_canon = _canonical_triangles(rust_lnas)
    np.testing.assert_allclose(
        py_canon, rust_canon, atol=1e-4,
        err_msg="Triangle vertex positions differ between Python and Rust",
    )

    # Same surface names and triangle counts per surface
    assert set(py_lnas.surfaces.keys()) == set(rust_lnas.surfaces.keys()), (
        f"Surface names differ: py={set(py_lnas.surfaces.keys())}, "
        f"rust={set(rust_lnas.surfaces.keys())}"
    )
    for name in py_lnas.surfaces:
        assert len(py_lnas.surfaces[name]) == len(rust_lnas.surfaces[name]), (
            f"Surface '{name}' triangle count differs: "
            f"py={len(py_lnas.surfaces[name])}, rust={len(rust_lnas.surfaces[name])}"
        )
