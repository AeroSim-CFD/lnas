import io
import unittest

import numpy as np

from lnas.stl import read_stl, stl_binary


class TestSTL(unittest.TestCase):
    def test_stl_pipeline(self):
        triangles = np.array(
            [[(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(1, 1, 0), (1, 0, 0), (0, 1, 0)]],
            dtype=np.float32,
        )
        normals = np.array([(0, 0, 1), (0, 0, 1)], dtype=np.float32)

        buff = stl_binary(triangles, normals)
        ret_triangles, ret_normals = read_stl(io.BytesIO(buff))

        np.testing.assert_equal(ret_triangles, triangles)
        np.testing.assert_equal(ret_normals, normals)


if __name__ == "__main__":
    unittest.main()
