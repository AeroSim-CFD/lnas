import io

import numpy as np

# One binary STL triangle record: a normal, three vertices and a 2-byte
# attribute field, all little-endian, packed into exactly 50 bytes. Used to
# read/write every triangle in a single vectorized NumPy operation.
_STL_RECORD_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attr", "<u2"),
    ]
)
# Attribute byte count padding the previous loop wrote as b"__".
_STL_ATTR_PADDING = 0x5F5F


def stl_binary(triangles: np.ndarray, normals: np.ndarray) -> bytes:
    """Binary representation of triangles and its normals in STL format

    Args:
        triangles (np.ndarray): Triangles to represent, shape must be (N, 3, 3)
        normals (np.ndarray): Triangles' normals, shape must be (N, 3)

    Returns:
        bytes: binary STL content
    """

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

    n_triangles = len(triangles)

    # 80 bytes header + uint32 triangle count + 50 bytes per triangle.
    header_bytes = b"Aerosim Exported STL".ljust(80, b"\00")
    count_bytes = np.array(n_triangles, dtype="<u4").tobytes()

    # Fill every triangle record in one shot instead of a per-triangle loop.
    records = np.empty(n_triangles, dtype=_STL_RECORD_DTYPE)
    records["normal"] = normals
    records["vertices"] = triangles
    records["attr"] = _STL_ATTR_PADDING

    return header_bytes + count_bytes + records.tobytes()


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
    n_triangles = int(np.frombuffer(buff.read(4), dtype="<u4")[0])
    if n_triangles == 0:
        raise ValueError("Unable to read number of triangles as 0")

    # Parse every 50-byte triangle record in a single vectorized read.
    records = np.frombuffer(
        buff.read(50 * n_triangles), dtype=_STL_RECORD_DTYPE, count=n_triangles
    )

    # astype yields writable, contiguous copies (frombuffer is read-only).
    triangles = records["vertices"].astype(np.float32)
    normals = records["normal"].astype(np.float32)

    return triangles, normals
