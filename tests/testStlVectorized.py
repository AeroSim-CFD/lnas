import io
import pathlib

import numpy as np

from lnas import LnasFormat
from lnas.stl import read_stl, stl_binary


def _stl_binary_loop(triangles: np.ndarray, normals: np.ndarray) -> bytes:
    """Reference per-triangle loop implementation, used for parity checks."""
    header_bytes = b"Aerosim Exported STL".ljust(80, b"\00")
    n_triangles = len(triangles)
    stl_content = bytearray(len(header_bytes) + 4 + 50 * n_triangles)
    stl_content[: len(header_bytes)] = header_bytes
    stl_content[len(header_bytes) : len(header_bytes) + 4] = np.uint32(n_triangles).tobytes()
    idx_curr = len(header_bytes) + 4
    for idx, t_verts in enumerate(triangles):
        p0, p1, p2 = [t_verts[0], t_verts[1], t_verts[2]]
        normal = normals[idx]
        t_arr = np.array([normal, p0, p1, p2], dtype=np.float32)
        stl_content[idx_curr : idx_curr + 50] = t_arr.tobytes("C") + b"__"
        idx_curr += 50
    return bytes(stl_content)


def _read_stl_loop(buff: io.BytesIO) -> tuple[np.ndarray, np.ndarray]:
    """Reference per-triangle loop reader, used for parity checks."""
    buff.read(80)
    n_triangles = np.frombuffer(buff.read(4), dtype=np.uint32)[0]
    triangles = np.empty((n_triangles, 3, 3), dtype=np.float32)
    normals = np.empty((n_triangles, 3), dtype=np.float32)
    for idx in range(n_triangles):
        content = buff.read(50)
        normals[idx] = np.frombuffer(content[0:12], dtype=np.float32)
        triangles[idx] = np.frombuffer(content[12:48], dtype=np.float32).reshape((3, 3))
    return triangles, normals


def _sample_mesh() -> tuple[np.ndarray, np.ndarray]:
    geometry = LnasFormat.from_file(pathlib.Path("fixture/cylinder.lnas")).geometry
    return geometry.triangle_vertices.astype(np.float32), geometry.normals.astype(np.float32)


def test_stl_binary_byte_identical_to_loop():
    triangles, normals = _sample_mesh()
    assert stl_binary(triangles, normals) == _stl_binary_loop(triangles, normals)


def test_read_stl_parity_with_loop():
    triangles, normals = _sample_mesh()
    buff = stl_binary(triangles, normals)

    tri_vec, nrm_vec = read_stl(io.BytesIO(buff))
    tri_loop, nrm_loop = _read_stl_loop(io.BytesIO(buff))

    np.testing.assert_array_equal(tri_vec, tri_loop)
    np.testing.assert_array_equal(nrm_vec, nrm_loop)


def test_stl_roundtrip_preserves_values():
    triangles, normals = _sample_mesh()
    tri_out, nrm_out = read_stl(io.BytesIO(stl_binary(triangles, normals)))

    np.testing.assert_array_equal(tri_out, triangles)
    np.testing.assert_array_equal(nrm_out, normals)
    # frombuffer output is read-only; downstream code mutates these, so the
    # returned arrays must be writable.
    assert tri_out.flags.writeable
    assert nrm_out.flags.writeable
