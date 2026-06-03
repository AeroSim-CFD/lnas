import pathlib

import numpy as np

from lnas import LnasFormat, LnasGeometry


def _reference_normals_areas(geometry: LnasGeometry):
    """Independent (pre-sharing) reference: recompute the cross product twice,
    exactly as the previous implementation did, for a parity check."""
    tv = geometry.triangle_vertices
    u = tv[:, 1, :] - tv[:, 0, :]
    v = tv[:, 2, :] - tv[:, 0, :]
    cross = np.cross(u, v)
    normals = cross / np.linalg.norm(cross, axis=1)[:, np.newaxis]
    areas = np.linalg.norm(cross, axis=1) / 2
    return normals, areas


def test_full_update_shares_cross_product_parity():
    cylinder = LnasFormat.from_file(pathlib.Path("fixture/cylinder.lnas"))
    geometry = cylinder.geometry

    ref_normals, ref_areas = _reference_normals_areas(geometry)
    np.testing.assert_allclose(geometry.normals, ref_normals, atol=1e-6)
    np.testing.assert_allclose(geometry.areas, ref_areas, atol=1e-6)


def test_full_update_parity_after_transformation():
    from lnas import TransformationsMatrix

    cylinder = LnasFormat.from_file(pathlib.Path("fixture/cylinder.lnas"))
    geometry = cylinder.geometry
    geometry.apply_transformation(TransformationsMatrix(scale=(1, 2, 3), translation=(1, 4, 2)))

    ref_normals, ref_areas = _reference_normals_areas(geometry)
    np.testing.assert_allclose(geometry.normals, ref_normals, atol=1e-6)
    np.testing.assert_allclose(geometry.areas, ref_areas, atol=1e-6)


def test_remove_invalid_normals_drops_degenerate_triangle():
    # Two triangles: the first is valid, the second is degenerate (collinear
    # points -> zero cross product -> NaN normal), which must be dropped.
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.uint32)
    geometry = LnasGeometry(vertices=vertices, triangles=triangles)

    geometry._full_update(remove_invalid_normals=True)

    assert len(geometry.triangles) == 1
    assert not np.isnan(geometry.normals).any()
    assert not np.isnan(geometry.areas).any()
    # normals, areas and vertices_normals stay length-consistent after removal
    assert geometry.normals.shape[0] == 1
    assert geometry.areas.shape[0] == 1
    assert geometry.vertices_normals.shape[0] == len(geometry.vertices)
