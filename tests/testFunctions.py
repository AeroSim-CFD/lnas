import unittest

import numpy as np
from lnas import LnasFormat, LnasGeometry
from lnas.functions import combine_geometries, combine_lnas, filter_from_list


class TestFunctions(unittest.TestCase):
    def setUp(self):
        vertices = np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]])
        triangles = np.array([[0, 1, 2], [1, 3, 2]])
        geometry = LnasGeometry(vertices=vertices, triangles=triangles)
        self.mesh = LnasFormat(
            version="",
            geometry=geometry,
            surfaces={"sfc1": np.array([0]), "sfc2": np.array([1])},
        )

    def test_no_excluded_surfaces(self):
        sfc_list = []
        with self.assertRaises(Exception) as context:
            filter_from_list(surface_list=sfc_list, fmt=self.mesh)
        self.assertEqual(
            str(context.exception), "No geometry could be filtered from the list of surfaces."
        )

    def test_excluded_surface_not_in_mesh(self):
        sfc_list = ["sfc3"]
        with self.assertRaises(KeyError) as context:
            filter_from_list(surface_list=sfc_list, fmt=self.mesh)

        self.assertTrue(len(str(context.exception)) > 0)

    def test_filter_geometry_from_list(self):
        sfc_list = ["sfc1"]

        geometry, triangle_idx = filter_from_list(surface_list=sfc_list, fmt=self.mesh)

        self.assertIsInstance(geometry, LnasGeometry)
        self.assertIsInstance(triangle_idx, np.ndarray)

        self.assertTrue(triangle_idx == np.array([0]))

    def test_combine_geometries(self):
        geometry_list = [
            LnasGeometry(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                triangles=np.array([[0, 1, 2]]),
            ),
            LnasGeometry(
                vertices=np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]]),
                triangles=np.array([[0, 1, 2]]),
            ),
        ]

        geometry = combine_geometries(geometry_list)

        self.assertIsInstance(geometry, LnasGeometry)
        self.assertTrue((geometry.triangles == np.array([[3, 4, 5], [0, 1, 2]])).all())
        self.assertTrue(len(geometry.vertices) == 6)

    def test_combine_format(self):
        # Should use lnas as fixtures or a mock geometry?
        ...


if __name__ == "__main__":
    unittest.main()
