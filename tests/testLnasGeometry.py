import pathlib
import unittest

import numpy as np

from lnas import LnasFormat, LnasGeometry


class TestLnasGeometry(unittest.TestCase):
    def test_join_geometries(self):
        vertices = np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]])
        triangles = np.array([[0, 1, 2], [1, 3, 2]])
        geometry = LnasGeometry(vertices=vertices, triangles=triangles)
        other_geometry = geometry.copy()
        other_geometry.vertices[:, 2] += 10

        geometry = geometry.copy()
        geometry.join([other_geometry])

        expected_tri = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]])

        self.assertIsInstance(geometry, LnasGeometry)
        self.assertTrue((geometry.triangles == expected_tri).all())
        self.assertTrue(len(geometry.vertices) == 8)

    def test_geometry_normal_and_area(self):
        verts_pos = vp = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=np.float32)
        # Normal positive and normal negative
        triangles = np.array([(0, 1, 2), (0, 2, 1)], dtype=np.uint32)

        geometry = LnasGeometry(vertices=verts_pos, triangles=triangles)
        triangle_points = geometry.triangle_vertices
        normals = geometry.normals
        areas = geometry.areas

        np.testing.assert_almost_equal(
            triangle_points, [[vp[0], vp[1], vp[2]], [vp[0], vp[2], vp[1]]]
        )
        np.testing.assert_almost_equal(normals, [(0, 0, 1), (0, 0, -1)])
        np.testing.assert_almost_equal(areas, [0.5, 0.5])

    def test_geometry_save_stl(self):
        verts_pos = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=np.float32)
        # Normal positive and normal negative
        triangles = np.array([(0, 1, 2), (0, 2, 1)], dtype=np.uint32)

        geometry = LnasGeometry(vertices=verts_pos, triangles=triangles)
        filename = pathlib.Path("output/lagr.stl")
        geometry.export_stl(filename)

    def test_geometry_filter_triangles_simple(self):
        vertices = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 10], [0, 1, 10], [1, 0, 10]]
        triangles = [[0, 1, 2], [3, 4, 5]]
        geometry = LnasGeometry(
            vertices=np.array(vertices, dtype=np.float32),
            triangles=np.array(triangles, dtype=np.uint32),
        )
        start, end = (0, 0, 0), (5, 5, 5)

        triangles_filtered = geometry.triangles_inside_volume(start, end)
        for t_idx, t in enumerate(geometry.triangles):
            is_in = False
            for v_idx in t:
                v = geometry.vertices[v_idx]
                if (v >= start).all() and (v <= end).all():
                    self.assertTrue(triangles_filtered[t_idx])
                    is_in = True
                    break
            if not is_in:
                self.assertFalse(triangles_filtered[t_idx])

    def test_geometry_filter_triangles_cylinder(self):
        filename = pathlib.Path("fixture/cylinder.lnas")
        cylinder = LnasFormat.from_file(filename)

        start, end = (0, 0, 0), (2, 5, 1)
        geometry = cylinder.geometry
        triangles_filtered = geometry.triangles_inside_volume(start, end)
        for t_idx, t in enumerate(geometry.triangles):
            is_in = False
            for v_idx in t:
                v = geometry.vertices[v_idx]
                if (v >= start).all() and (v <= end).all():
                    self.assertTrue(triangles_filtered[t_idx])
                    is_in = True
                    break
            if not is_in:
                self.assertFalse(triangles_filtered[t_idx])


if __name__ == "__main__":
    unittest.main()
