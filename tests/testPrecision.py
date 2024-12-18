import pathlib

import numpy as np

from lnas import LnasFormat
from lnas.transformations import TransformationsMatrix


def to_np_arr(array):
    return np.array(array, dtype="float32")


def test_precision_transform_stl():
    """Test from issue #9

      - transformation:
          translation: [-3569.5809326171875, -306.52301025390625, 0.0]
          fixed_point: [0,0,0]
          rotation: [0,0,0]
          scale: [1,1,1]
    - transformation:
          translation: [0, 0, 0]
          fixed_point: [0,0,0]
          rotation: [0,0,1.3264502315156903]
          scale: [1,1,1]
    - transformation:
          translation: [0, 0, 0]
          fixed_point: [0,0,0]
          rotation: [0,0,0]
          scale: [0.0625,0.0625,0.0625]
    """
    lnas_filename = pathlib.Path("fixture/Aero_Sim_HDB_v1.lnas")
    lnas_fmt = LnasFormat.from_file(lnas_filename)

    t1 = TransformationsMatrix(
        translation=to_np_arr((-3569.5809326171875, -306.52301025390625, 0.0))
    )
    t2 = TransformationsMatrix(angle=to_np_arr((0, 0, 1.3264502315156903)))
    t3 = TransformationsMatrix(scale=to_np_arr((0.00625, 0.00625, 0.00625)))

    print(lnas_fmt.geometry.triangle_vertices.dtype)
    assert not np.any(np.isnan(lnas_fmt.geometry.triangle_vertices))
    for t in (t1, t2, t3):
        print("transformation", t)
        lnas_fmt.geometry.apply_transformation(t)
        lnas_fmt.geometry._full_update()
    assert not np.any(np.isnan(lnas_fmt.geometry.triangle_vertices))
    assert not np.any(np.isnan(lnas_fmt.geometry.areas))
    assert not np.any(np.isnan(lnas_fmt.geometry.normals))

    filename = pathlib.Path("output/precision_test.stl")
    lnas_fmt.export_stl(filename)
