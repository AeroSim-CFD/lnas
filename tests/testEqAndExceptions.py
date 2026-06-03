import numpy as np
import pytest

from lnas import LnasFormat, LnasGeometry
from lnas.exceptions import LnasVersionError
from lnas.transformations import apply_transformation_matrix


def _simple_geometry() -> LnasGeometry:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.uint32)
    return LnasGeometry(vertices=vertices, triangles=triangles)


def test_geometry_eq_with_non_geometry_returns_false():
    geometry = _simple_geometry()
    # Comparing against unrelated objects must return False, not raise AttributeError.
    assert (geometry == object()) is False
    assert (geometry == None) is False  # noqa: E711
    assert (geometry == 42) is False
    assert geometry == geometry.copy()


def test_format_eq_with_non_format_returns_false():
    lnas = LnasFormat(version="", geometry=_simple_geometry(), surfaces={})
    assert (lnas == object()) is False
    assert (lnas == None) is False  # noqa: E711
    assert lnas == lnas.copy()


def test_version_error_is_exception():
    # Must subclass Exception so normal `except Exception` handlers catch it.
    assert issubclass(LnasVersionError, Exception)
    with pytest.raises(Exception):
        raise LnasVersionError("boom")


def test_from_dct_rejects_unsupported_version():
    dct = {"version": "v9.9.9", "geometry": {}, "surfaces": {}}
    with pytest.raises(LnasVersionError):
        LnasFormat.from_dct(dct)


def test_apply_transformation_matrix_rejects_invalid_arr_type():
    arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    M = np.identity(4, dtype=np.float32)
    with pytest.raises(ValueError):
        apply_transformation_matrix(arr, M, arr_type="bogus")  # type: ignore[arg-type]
