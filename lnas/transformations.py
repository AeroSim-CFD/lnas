from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = ["Transformations", "TransformationsMatrix"]


class Transformations:
    """Class to get matrixes to perform transformations, as translating, scaling, etc."""

    @classmethod
    def get_rotation_angles_normal_to_z(cls, n: np.ndarray) -> np.ndarray:
        """Angles to rotate an normal to positive z-axis"""
        # project n1 in x
        d = (n[1] ** 2 + n[2] ** 2) ** 0.5
        # cosx = n[2] / d
        # sinx = n[1] / d
        x_angle = np.arctan2(n[1], n[2])
        cosy = d
        siny = n[0]
        y_angle = np.arctan2(siny, cosy)
        z_angle = 0
        return np.array((x_angle, y_angle, z_angle), dtype="float32")

    @classmethod
    def get_translation(cls, translation: np.ndarray) -> np.ndarray:
        m_trans = np.identity(4, dtype="float32")
        m_trans[0:3, 3] = translation
        return m_trans

    @classmethod
    def get_rotation_x(cls, angle: float) -> np.ndarray:
        m_rot_x = np.identity(4, dtype="float32")
        acos = np.cos(angle)
        asin = np.sin(angle)
        m_rot_x[1, 1] = acos
        m_rot_x[1, 2] = -asin
        m_rot_x[2, 1] = asin
        m_rot_x[2, 2] = acos
        return m_rot_x

    @classmethod
    def get_rotation_y(cls, angle: float) -> np.ndarray:
        m_rot_y = np.identity(4, dtype="float32")
        acos = np.cos(angle)
        asin = np.sin(angle)
        m_rot_y[0, 0] = acos
        m_rot_y[0, 2] = asin
        m_rot_y[2, 0] = -asin
        m_rot_y[2, 2] = acos
        return m_rot_y

    @classmethod
    def get_rotation_z(cls, angle: float) -> np.ndarray:
        m_rot_z = np.identity(4, dtype="float32")
        acos = np.cos(angle)
        asin = np.sin(angle)
        m_rot_z[0, 0] = acos
        m_rot_z[0, 1] = -asin
        m_rot_z[1, 0] = asin
        m_rot_z[1, 1] = acos
        return m_rot_z

    @classmethod
    def get_scale(cls, scale: tuple[float, float, float]) -> np.ndarray:
        m_scale = np.identity(4, dtype="float32")
        m_scale[1, 1] = scale[1]
        m_scale[0, 0] = scale[0]
        m_scale[2, 2] = scale[2]
        return m_scale

    @classmethod
    def get_centering_matrix(self, fixed_point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m_go_rot_center = np.identity(4, dtype="float32")
        m_back_rot_center = np.identity(4, dtype="float32")
        m_go_rot_center[0:3, 3] = -fixed_point
        m_back_rot_center[0:3, 3] = fixed_point
        return (m_go_rot_center, m_back_rot_center)


@dataclass
class TransformationsMatrix:
    """Represent the transformation matrix and its informations"""

    # Angle to rotate in each axis in radians (right hand rule)
    angle: np.ndarray = field(default_factory=lambda: np.array((0, 0, 0), dtype="float32"))
    # Translation values
    translation: np.ndarray = field(default_factory=lambda: np.array((0, 0, 0), dtype="float32"))
    # Scale values
    scale: np.ndarray = field(default_factory=lambda: np.array((1, 1, 1), dtype="float32"))
    # Fixed point used as reference to rotate and scale (usually geometry centroid)
    fixed_point: np.ndarray = field(default_factory=lambda: np.array((0, 0, 0), dtype="float32"))
    # Always update matrixes
    always_update: bool = True

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.angle),
                tuple(self.translation),
                tuple(self.fixed_point),
                tuple(self.scale),
            )
        )

    def __post_init__(self):
        # matrixes to operate (pre create all)
        # translation
        self.m_trans = np.identity(4, dtype="float32")
        # scaling
        self.m_scale = np.identity(4, dtype="float32")
        # center object (to rotate, etc)
        self.m_go_rot_center = np.identity(4, dtype="float32")
        self.m_back_rot_center = np.identity(4, dtype="float32")
        # rotations in each axis
        self.m_rot_x = np.identity(4, dtype="float32")
        self.m_rot_y = np.identity(4, dtype="float32")
        self.m_rot_z = np.identity(4, dtype="float32")
        self.update_all()

    def update_all(self):
        """Updates all transformation partial matrices"""
        self.m_trans = Transformations.get_translation(self.translation)
        self.m_rot_x = Transformations.get_rotation_x(self.angle[0])
        self.m_rot_y = Transformations.get_rotation_y(self.angle[1])
        self.m_rot_z = Transformations.get_rotation_z(self.angle[2])
        self.m_scale = Transformations.get_scale(self.scale)
        (
            self.m_go_rot_center,
            self.m_back_rot_center,
        ) = Transformations.get_centering_matrix(self.fixed_point)

    @property
    def m_rotation_full(self) -> np.ndarray:
        M = np.matmul(self.m_rot_x, self.m_rot_y)
        return np.matmul(M, self.m_rot_z)

    @property
    def m_rotation_full_inv(self) -> np.ndarray:
        M = np.matmul(self.m_rot_z, self.m_rot_y)
        return np.matmul(M, self.m_rot_x)

    @property
    def transformation_matrix(self) -> np.ndarray:
        """Transformation matrix to scale, rotate and translate the object

        Returns:
            np.ndarray: tranformation matrix (as in OpenGL)
        """

        if self.always_update:
            self.update_all()

        # first centralize, then scale, rotate, translate, decentralize
        order = [
            self.m_go_rot_center,
            self.m_scale,
            self.m_rotation_full,
            self.m_trans,
            self.m_back_rot_center,
        ]
        M = order[0]
        for o in order[1:]:
            M = np.matmul(o, M)
        return M

    def apply(
        self, arr: np.ndarray, arr_type: Literal["point", "vector"], invert_transf: bool = False
    ) -> np.ndarray:
        M = self.transformation_matrix
        return apply_transformation_matrix(arr, M, arr_type, invert_transf)

    def apply_points(self, arr: np.ndarray, invert_transf: bool = False) -> np.ndarray:
        return self.apply(arr, arr_type="point", invert_transf=invert_transf)

    def apply_vectors(self, arr: np.ndarray, invert_transf: bool = False) -> np.ndarray:
        return self.apply(arr, arr_type="vector", invert_transf=invert_transf)


def apply_transformation_matrix(
    arr: np.ndarray,
    M: np.ndarray,
    arr_type: Literal["point", "vector"],
    invert_transf: bool = False,
) -> np.ndarray:
    """Apply transformation matrix T to array of points arr

    Args:
        arr (np.ndarray): Array to apply transformation, shaped as [N, 3]
        M (np.ndarray): Transformation to apply, shaped as [4, 4]. Same logic as OpenGL
        arr_type (Literal["point", "vector"]): Array type to consider, "point" applies translation as well, "vector" doesn't
        invert_transf (bool, optional): Invert transformation matrix before applying. Defaults to False.

    Returns:
        np.ndarray: Points transformed
    """

    if arr.shape[1] != 3:
        raise ValueError("Array points must be 3D to be transformed")
    # Point transformation appends 1 (has translation), vector appends 0 (no translation)
    if arr_type == "point":
        col_add = np.ones((arr.shape[0], 1), dtype=arr.dtype)
    elif arr_type == "vector":
        col_add = np.zeros((arr.shape[0], 1), dtype=arr.dtype)

    if invert_transf:
        M = np.linalg.inv(M)

    # Add column to operate transformation
    arr_transf = np.append(arr, col_add, axis=1)
    # Apply transformation
    arr_transf = np.matmul(M, arr_transf.T)
    # Remove 0 or 1 added
    arr_transf = arr_transf.T[:, :3]
    return arr_transf
