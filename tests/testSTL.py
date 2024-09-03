import io

import numpy as np
import pytest

from lnas.stl import read_stl, stl_binary


@pytest.fixture()
def triangles():
    yield np.array(
        [[(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(1, 1, 0), (1, 0, 0), (0, 1, 0)]],
        dtype=np.float32,
    )


@pytest.fixture()
def normals():
    yield np.array([(0, 0, 1), (0, 0, 1)], dtype=np.float32)


def test_stl_pipeline(triangles, normals):
    buff = stl_binary(triangles, normals)
    ret_triangles, ret_normals = read_stl(io.BytesIO(buff))

    np.testing.assert_equal(ret_triangles, triangles)
    np.testing.assert_equal(ret_normals, normals)
