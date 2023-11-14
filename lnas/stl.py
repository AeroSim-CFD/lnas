import io

import numpy as np


def stl_binary(triangles: np.ndarray, normals: np.ndarray) -> bytes:
    """Binary representation of triangles and its normals in STL format

    Args:
        triangles (np.ndarray): Triangles to represent, shape must be (N, 3, 3)
        normals (np.ndarray): Triangles' normals, shape must be (N, 3)

    Returns:
        bytes: binary STL content
    """

    if len(triangles) == 0:
        raise ValueError("There must be triangles to export a STL")
    if len(triangles.shape) != 3 or triangles.shape[1] != 3 or triangles.shape[2] != 3:
        raise ValueError(f"Triangles shape must be (N, 3, 3) for STL. Shape is {triangles.shape}")
    if len(normals.shape) != 2 or normals.shape[1] != 3:
        raise ValueError(
            f"Triangles normals shape must be (N, 3) for STL. Shape is {normals.shape}"
        )
    if normals.shape[0] != triangles.shape[0]:
        raise ValueError(
            f"Normal and triangles must have same N. triangles: {triangles}, normals: {normals}"
        )

    # 80 bytes header
    header_str = b"Aerosim Exported STL"
    header_bytes = header_str.ljust(80, b"\00")

    n_triangles = len(triangles)
    # STL content is header + uint32 + 50 bytes per triangle
    stl_content = bytearray(len(header_bytes) + 4 + 50 * n_triangles)

    # Add header and number of triangles
    stl_content[: len(header_bytes)] = header_bytes
    stl_content[len(header_bytes) : len(header_bytes) + 4] = np.uint32(n_triangles).tobytes()

    idx_curr = len(header_bytes) + 4

    for idx, t_verts in enumerate(triangles):
        p0, p1, p2 = [t_verts[0], t_verts[1], t_verts[2]]
        normal = normals[idx]
        t_arr = np.array([normal, p0, p1, p2], dtype=np.float32)

        # Add triangles to STL content and 2 bytes padding
        stl_content[idx_curr : idx_curr + 50] = t_arr.tobytes("C") + b"__"
        idx_curr += 50
    return stl_content


def read_stl(buff: io.BytesIO) -> tuple[np.ndarray, np.ndarray]:
    """Read buffer content as STL file

    Args:
        buff (io.BufferedReader): buffer to read from

    Returns:
        tuple[np.ndarray, np.ndarray]: return STL representation as (triangles, normals).
    """

    # pass header
    buff.read(80)
    # Read number of triangles

    n_triangles = np.frombuffer(buff.read(4), dtype=np.uint32)[0]
    if n_triangles == 0:
        raise ValueError("Unable to read number of triangles as 0")

    triangles = np.empty((n_triangles, 3, 3), dtype=np.float32)
    normals = np.empty((n_triangles, 3), dtype=np.float32)

    for idx in range(n_triangles):
        content = buff.read(50)
        normal = np.frombuffer(content[0:12], dtype=np.float32)
        triangle = np.frombuffer(content[12:48], dtype=np.float32).reshape((3, 3))
        triangles[idx] = triangle
        normals[idx] = normal

    return triangles, normals
