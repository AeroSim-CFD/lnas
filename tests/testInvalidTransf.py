import pathlib

import numpy as np
import pytest

from lnas import LnasFormat, LnasGeometry
from lnas.transformations import TransformationsMatrix

files = ["invalid_normals_after_transf.stl"]
transf = TransformationsMatrix.from_tuple(
    scale=(1 / 16, 1 / 16, 1 / 16),
    translation=(432 / 16, 448 / 16, 64 / 16),
)


def test_invalid_transf():
    """Original fixture-driven regression. The fixture is optional; the
    inline tests below always exercise the invalid-normal removal path."""
    for f in files:
        p = pathlib.Path("fixture/stl_invalid_transf/") / f
        if not p.exists():
            continue
        fmt = LnasFormat.from_stl(p)
        fmt.geometry.apply_transformation(transf, remove_invalid_normals=True)


def _mesh_with_degenerate_triangle() -> LnasGeometry:
    # Vertices 0-2 form a valid triangle; vertices 0,1,3 are collinear, so the
    # second triangle is degenerate and produces a NaN normal.
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.uint32)
    return LnasGeometry(vertices=vertices, triangles=triangles)


def test_apply_transformation_removes_invalid_normals():
    geometry = _mesh_with_degenerate_triangle()
    geometry.apply_transformation(transf, remove_invalid_normals=True)

    # The degenerate triangle is dropped, the valid one survives, and no NaNs
    # leak into the derived quantities.
    assert len(geometry.triangles) == 1
    assert not np.isnan(geometry.normals).any()
    assert not np.isnan(geometry.areas).any()
    assert not np.isnan(geometry.vertices_normals).any()


def test_apply_transformation_raises_when_not_removing():
    geometry = _mesh_with_degenerate_triangle()
    with pytest.raises(ValueError):
        geometry.apply_transformation(transf, remove_invalid_normals=False)
