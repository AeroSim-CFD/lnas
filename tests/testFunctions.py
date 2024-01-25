import unittest

import numpy as np
from lnas import LnasFormat, LnasGeometry
from lnas.functions import combine_geometries, combine_lnas, filter_from_list


class TestFunctions(unittest.TestCase):
    def setUp(self):
        vertices = np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]])
        triangles = np.array([[0, 1, 2], [1, 3, 2]])
        geometry = LnasGeometry(vertices=vertices, triangles=triangles)
        other_geometry = geometry.copy()
        other_geometry.vertices[:, 2] += 10
        self.mesh = LnasFormat(
            version="",
            geometry=geometry,
            surfaces={"sfc1": np.array([0]), "sfc2": np.array([1])},
        )
        self.other_mesh = LnasFormat(
            version="",
            geometry=other_geometry,
            surfaces={"sfc1": np.array([0]), "sfc2": np.array([1])},
        )
        self.other_mesh_same_surfaces = LnasFormat(
            version="",
            geometry=other_geometry,
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
        geometry_list = [self.mesh.geometry, self.other_mesh.geometry]
        geometry = combine_geometries(geometry_list)
        expected_tri = np.array([[4, 5, 6], [5, 7, 6], [0, 1, 2], [1, 3, 2]])

        self.assertIsInstance(geometry, LnasGeometry)
        self.assertTrue((geometry.triangles == expected_tri).all())
        self.assertTrue(len(geometry.vertices) == 8)

    def test_combine_format(self):
        combined_lnas = combine_lnas(
            lnas_fmts=[self.mesh, self.other_mesh], surfaces_suffixes=["_sfc1", "_sfc2"]
        )
        expected_tri = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]])
        expected_sfcs = [k + "_sfc1" for k in self.mesh.surfaces.keys()] + [
            k + "_sfc2" for k in self.mesh.surfaces.keys()
        ]

        self.assertIsInstance(combined_lnas, LnasFormat)
        self.assertEqual(len(combined_lnas.geometry.vertices), 8)
        self.assertTrue((combined_lnas.geometry.triangles == expected_tri).all())
        self.assertTrue(list(combined_lnas.surfaces.keys()) == expected_sfcs)


if __name__ == "__main__":
    unittest.main()
