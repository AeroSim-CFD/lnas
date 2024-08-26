import unittest

import numpy as np

from lnas import TransformationsMatrix


class TestTransformationMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def to_np_arr(_, array):
            return np.array(array, dtype="float32")

        cls.to_np_arr = to_np_arr

    def test_geometry_translation(self):
        point = self.to_np_arr([1, 1, 1, 1])
        m = TransformationsMatrix(translation=self.to_np_arr([0, 1, -2]))
        T = m.transformation_matrix
        p = np.matmul(T, point)
        np.testing.assert_almost_equal(p[:3], (1, 2, -1), decimal=5)

    def test_geometry_scale(self):
        vector = self.to_np_arr([1, 1, 1, 1])
        m = TransformationsMatrix(scale=self.to_np_arr([0, 1, -2]))
        T = m.transformation_matrix
        p = np.matmul(T, vector)
        np.testing.assert_almost_equal(p[:3], (0, 1, -2), decimal=5)

    def test_geometry_rotation_z(self):
        point = self.to_np_arr([2, 1, 1, 1])
        m = TransformationsMatrix(
            angle=self.to_np_arr([0, 0, np.pi / 2]),
            fixed_point=self.to_np_arr([1, 1, 1]),
        )
        T = m.transformation_matrix
        p = np.matmul(T, point)
        np.testing.assert_almost_equal(p[:3], (1, 2, 1), decimal=5)

    def test_geometry_rotation_y(self):
        point = self.to_np_arr([1, 0, 0, 1])
        m = TransformationsMatrix(angle=self.to_np_arr([0, np.pi / 2, 0]))
        T = m.transformation_matrix
        p = np.matmul(T, point)
        # Right hand rule
        np.testing.assert_almost_equal(p[:3], (0, 0, -1), decimal=5)

    def test_geometry_rotation_x(self):
        point = self.to_np_arr([0, 1, 0, 1])
        m = TransformationsMatrix(angle=self.to_np_arr([np.pi / 2, 0, 0]))
        T = m.transformation_matrix
        p = np.matmul(T, point)
        # Right hand rule
        np.testing.assert_almost_equal(p[:3], (0, 0, 1), decimal=5)

    def test_apply_transformation_point(self):
        points = np.array([[1, 1, 1], [2, 1.5, 0.5]])
        m = TransformationsMatrix(translation=np.array([1, 2, 3]))
        points_transf = m.apply_points(points)
        # Right hand rule
        np.testing.assert_almost_equal(points_transf, [[2, 3, 4], [3, 3.5, 3.5]], decimal=5)

    def test_apply_transformation_vector(self):
        vecs = np.array([[1, 1, 1], [2, 1.5, 0.5]])
        m = TransformationsMatrix(scale=np.array([1.5, 2, 0.5]), translation=np.array([1, 2, 3]))
        vecs_transf = m.apply_vectors(vecs)
        # Right hand rule
        np.testing.assert_almost_equal(vecs_transf, [[1.5, 2, 0.5], [3, 3, 0.25]], decimal=5)

    def test_apply_invert_transformation(self):
        arr = np.array([[1, 1, 1], [2, 1.5, 0.5], [-1, 0, 3]])
        m = TransformationsMatrix(
            scale=np.array([1.5, 2, 0.5]),
            translation=np.array([1, 2, 3]),
            angle=np.array([0.1, 0.2, -0.5]),
            fixed_point=np.array([10, 1, -5]),
        )

        for v_type in ("point", "vector"):
            tm = m.apply(arr, v_type, invert_transf=False)
            t = m.apply(tm, v_type, invert_transf=True)
            np.testing.assert_almost_equal(t, arr, decimal=5)


if __name__ == "__main__":
    unittest.main()
