import pathlib

from lnas import LnasFormat
from lnas.transformations import TransformationsMatrix

files = ["invalid_normals_after_transf.stl"]
transf = TransformationsMatrix.from_tuple(
    scale=(1 / 16, 1 / 16, 1 / 16),
    translation=(432 / 16, 448 / 16, 64 / 16),
)


def test_invalid_transf():
    for f in files:
        p = pathlib.Path("fixture/stl_invalid_transf/") / f
        if(not p.exists()):
            continue
        fmt = LnasFormat.from_stl(p)
        fmt.geometry.apply_transformation(transf, remove_invalid_normals=True)
