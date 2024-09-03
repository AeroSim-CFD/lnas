import itertools
import pathlib

import numpy as np
import pytest

from lnas import LnasFormat, LnasGeometry, TransformationsMatrix


@pytest.fixture()
def geometry():
    vertices = np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]])
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    yield LnasGeometry(vertices=vertices, triangles=triangles)


@pytest.fixture()
def other_geometry(geometry):
    other_geometry = geometry.copy()
    other_geometry.vertices[:, 2] += 10
    yield other_geometry


@pytest.fixture()
def mesh(geometry):
    yield LnasFormat(
        version="",
        geometry=geometry,
        surfaces={"sfc1": np.array([0]), "sfc2": np.array([1])},
    )


@pytest.fixture()
def other_mesh(other_geometry):
    yield LnasFormat(
        version="",
        geometry=other_geometry,
        surfaces={"sfc1": np.array([0]), "sfc2": np.array([1])},
    )


@pytest.fixture()
def other_mesh_same_surfaces(other_geometry):
    yield LnasFormat(
        version="",
        geometry=other_geometry,
        surfaces={"sfc1": np.array([0]), "sfc2": np.array([1])},
    )


def test_excluded_surface_not_in_mesh(mesh):
    sfc_list = ["sfc3"]
    with pytest.raises(KeyError) as context:
        mesh.geometry_from_list_surfaces(sfc_list)

    assert len(str(context.value)) > 0


def test_filter_surface_from_list(mesh):
    sfc_list = ["sfc1"]

    geometry, triangle_idx = mesh.geometry_from_list_surfaces(sfc_list)

    assert isinstance(geometry, LnasGeometry)
    assert isinstance(triangle_idx, np.ndarray)

    assert triangle_idx == np.array([0])


def test_join_lnas(mesh, other_mesh):
    combined_lnas = mesh.copy()
    combined_lnas.join([other_mesh], ["_sfc2"])
    expected_tri = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]])
    expected_sfcs = [k for k in mesh.surfaces.keys()] + [k + "_sfc2" for k in mesh.surfaces.keys()]

    assert isinstance(combined_lnas, LnasFormat)
    assert len(combined_lnas.geometry.vertices) == 8
    assert (combined_lnas.geometry.triangles == expected_tri).all()
    assert list(combined_lnas.surfaces.keys()) == expected_sfcs


def test_cube_lnas_reading():
    filename = pathlib.Path("fixture/cube.lnas")
    cube = LnasFormat.from_file(filename)

    geometry = cube.geometry
    assert len(geometry.vertices) == 8
    assert len(geometry.triangles) == 6 * 2

    norm_term = geometry.vertices[:, 0].max()
    # For all combinations of vertices in a cube
    for p0, p1, p2 in itertools.product(*[[0, 1]] * 3):
        lagr_vert = (p0 * norm_term, p1 * norm_term, p2 * norm_term)
        assert lagr_vert in geometry.vertices


def test_cube_lnas_from_surface():
    filename = pathlib.Path("fixture/cube.lnas")
    cube = LnasFormat.from_file(filename)

    geometry_surface = cube.geometry_from_surface("cube")
    assert geometry_surface == cube.geometry

    with pytest.raises(KeyError):
        cube.geometry_from_surface("not_surface")


def test_cube_lnas_transformation():
    filename = pathlib.Path("fixture/cube.lnas")
    cube = LnasFormat.from_file(filename)
    geometry = cube.geometry
    norm_val = geometry.vertices[:, 0].max()

    translation = (1, 4, 2)
    scale = (1, 2, 3)
    transformation = TransformationsMatrix(scale=scale, translation=translation)

    geometry.apply_transformation(transformation)
    min_pos = tuple([min(geometry.vertices[:, d]) for d in range(3)])
    max_pos = tuple([max(geometry.vertices[:, d]) for d in range(3)])

    assert min_pos == translation
    assert max_pos == tuple(translation[d] + scale[d] * norm_val for d in range(3))


def test_cube_triangles_normal():
    filename = pathlib.Path("fixture/cube.lnas")
    geometry_fmt = LnasFormat.from_file(filename)
    geometry = geometry_fmt.geometry

    translation = (1, 4, 2)
    scale = (1, 2, 3)
    transformation = TransformationsMatrix(scale=scale, translation=translation)

    triangle_normals_first = geometry.normals
    geometry.apply_transformation(transformation)
    triangle_normals_new = geometry.normals

    np.testing.assert_almost_equal(triangle_normals_new, triangle_normals_first)

    rotation = TransformationsMatrix(angle=(0, 1.5, 2))
    geometry.apply_transformation(rotation)
    triangle_normals_rot = geometry.normals

    assert not np.allclose(triangle_normals_rot, triangle_normals_first)


def check_save(lnas_filename: str | pathlib.Path):
    lnas_filename = pathlib.Path(lnas_filename)
    lnas_fmt = LnasFormat.from_file(lnas_filename)
    geometry = lnas_fmt.geometry

    translation = (1, 4, 2)
    scale = (1, 2, 3)
    transformation = TransformationsMatrix(scale=scale, translation=translation)
    geometry.apply_transformation(transformation)

    filename = pathlib.Path("output/test.lnas")
    lnas_fmt.to_file(filename)
    cylinder_load = LnasFormat.from_file(filename)

    assert lnas_fmt == cylinder_load
    cylinder_intact = LnasFormat.from_file(lnas_filename)
    assert cylinder_intact != cylinder_load


def test_filter_triangles():
    lnas_filename = "fixture/cylinder.lnas"
    lnas_filename = pathlib.Path(lnas_filename)
    lnas_fmt = LnasFormat.from_file(lnas_filename)

    n_triangles = len(lnas_fmt.geometry.triangles)
    triangles_arange = np.arange(n_triangles, dtype=np.uint32)
    even_triangles = np.extract(triangles_arange % 2 == 0, triangles_arange)
    odd_triangles = np.extract(triangles_arange % 2 == 1, triangles_arange)

    lnas_fmt.surfaces["even"] = even_triangles
    lnas_fmt.surfaces["odd"] = odd_triangles

    even_geometry = lnas_fmt.geometry_from_surface("even")
    odd_geometry = lnas_fmt.geometry_from_surface("odd")

    # Filter divisible by 3
    filter_triangles = np.array(
        [1 if idx % 3 == 0 else 0 for idx in range(n_triangles)], dtype=bool
    )

    filtered_lnas = lnas_fmt.filter_triangles(filter_triangles)
    for s, arr in filtered_lnas.surfaces.items():
        for v in arr:
            assert v >= 0
            assert v < len(filtered_lnas.geometry.triangles)

    new_even_geometry = filtered_lnas.geometry_from_surface("even")
    new_odd_geometry = filtered_lnas.geometry_from_surface("odd")

    for idx, t in enumerate(lnas_fmt.geometry.triangles):
        arrays_in = []
        arrays_not_in = []
        if idx % 2 == 0:
            arrays_not_in.append(new_odd_geometry.triangles)
            arrays_not_in.append(odd_geometry.triangles)
            arrays_in.append(even_geometry.triangles)
            if idx % 3 == 0:
                arrays_in.append(new_even_geometry.triangles)
            else:
                arrays_not_in.append(new_even_geometry.triangles)
        else:
            arrays_not_in.append(new_even_geometry.triangles)
            arrays_not_in.append(even_geometry.triangles)
            arrays_in.append(odd_geometry.triangles)
            if idx % 3 == 0:
                arrays_in.append(new_odd_geometry.triangles)
            else:
                arrays_not_in.append(new_odd_geometry.triangles)

        for arr in arrays_in:
            assert tuple(t) in [tuple(tt) for tt in arr]
        for arr in arrays_not_in:
            assert tuple(t) not in [tuple(tt) for tt in arr]


def test_cylinder_save():
    check_save("fixture/cylinder.lnas")


def test_cube_no_norm():
    check_save("fixture/cube_no_norm.lnas")


def test_export_stl():
    lnas_filename = pathlib.Path("fixture/cylinder.lnas")
    geometry = LnasFormat.from_file(lnas_filename)
    filename = pathlib.Path("output/test.stl")
    geometry.export_stl(filename)
