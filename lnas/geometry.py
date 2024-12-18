from __future__ import annotations

import base64
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from lnas import TransformationsMatrix
from lnas.stl import stl_binary
from lnas.transformations import apply_transformation_matrix


@dataclass
class LnasGeometry:
    """Lagrangian geometry representation"""

    # Vertices positions (shape is (Np, 3))
    vertices: np.ndarray
    # Triangles vertices indexes (shape is (Nt, 3))
    triangles: np.ndarray

    def __eq__(self, __o: object) -> bool:
        if not isinstance(self, type(__o)):
            return False
        if not np.allclose(self.vertices, __o.vertices):
            return False
        if not np.array_equal(self.triangles, __o.triangles):
            return False
        return True

    def _update_triangles_vertices(self):
        nt = self.triangles.shape[0]
        # Indexed as (idx_triangle, n_vert, vert_value)
        idxs_triangles = self.triangles.flatten(order="C")
        verts = self.vertices[idxs_triangles]

        self._triangles_vertices = verts.reshape(nt, 3, 3)

    def copy(self):
        vertices = self.vertices.copy()
        triangles = self.triangles.copy()
        geometry = LnasGeometry(vertices=vertices, triangles=triangles)
        return geometry

    @property
    def triangle_vertices(self):
        if not hasattr(self, "_triangles_vertices"):
            self._update_triangles_vertices()
        return self._triangles_vertices

    def _cross_prod(self):
        triangle_points = self.triangle_vertices

        # Same convention as OpenGL (right hand rule)
        # U = p1 - p0; V = p2 - p0
        # let u = pp1 - pp0;
        # let v = pp2 - pp0;
        # let mut order_normal = u.cross(v);
        U = triangle_points[:, 1, :] - triangle_points[:, 0, :]
        V = triangle_points[:, 2, :] - triangle_points[:, 0, :]
        return np.cross(U, V)

    def _update_normals(self):
        cross_prod = self._cross_prod()
        self._normals = cross_prod / np.linalg.norm(cross_prod, axis=1)[:, np.newaxis]
        if np.isnan(self._normals).any():
            raise ValueError("Invalid normals generated, there is a NaN value")

    def _update_areas(self):
        cross_prod = self._cross_prod()
        self._areas = np.linalg.norm(cross_prod, axis=1) / 2

    @property
    def normals(self) -> np.ndarray:
        if not hasattr(self, "_normals"):
            self._update_normals()
        return self._normals

    def _update_vertices_normals(self):
        normals, areas, triangles = self.normals, self.areas, self.triangles
        vertices_normals = np.zeros((len(self.vertices), 3), dtype=np.float32)

        # Add triangle normal to vertices, considering triangle area
        for normal, triangle, area in zip(normals, triangles, areas):
            for v_idx in triangle:
                vertices_normals[v_idx] += normal * area
        # Normalize normal to its norm
        norms = np.linalg.norm(vertices_normals, axis=1)
        # Check where no vertex was used, to avoid division by zero
        v_idxs_zero = (vertices_normals == 0).all(axis=1)
        for v_idx in np.where(v_idxs_zero):
            norms[v_idx] = 1

        self._vertices_normals = vertices_normals / np.expand_dims(norms, axis=1)

        if np.isnan(self._vertices_normals).any():
            raise ValueError("Invalid vertices normals generated, there is a NaN value")

    @property
    def vertices_normals(self) -> np.ndarray:
        if not hasattr(self, "_vertices_normals"):
            self._update_vertices_normals()
        return self._vertices_normals

    @property
    def areas(self) -> np.ndarray:
        if not hasattr(self, "_areas"):
            self._update_areas()
        return self._areas

    def _full_update(self):
        # ORDER IS IMPORTANT, one depends on the other
        self._update_triangles_vertices()
        self._update_normals()
        self._update_areas()
        self._update_vertices_normals()

    @classmethod
    def from_dct(cls, dct: dict[str, Any]) -> LnasGeometry:
        """Load lagrangian geometry from dictionary"""
        dct_use = {}
        for key, dtype_use, last_dim in [
            ("vertices", np.float32, 3),
            ("triangles", np.uint32, 3),
        ]:
            val_str = dct[key]
            val_b64_bytes = val_str.encode("ascii")
            val_bytes = base64.b64decode(val_b64_bytes)
            np_arr = np.frombuffer(val_bytes, dtype=dtype_use)
            # Reshape to right dimension
            np_arr = np_arr.reshape((len(np_arr) // last_dim, last_dim))
            dct_use[key] = np_arr

        return LnasGeometry(**dct_use)

    def to_dct(self) -> dict[str, Any]:
        """Get lagrangian geometry as dictionary"""

        def arr_to_string(arr_np: np.ndarray, dtype: np.dtype) -> str:
            arr_use = arr_np.astype(dtype=dtype)
            arr_bytes = arr_use.tobytes(order="C")
            arr_b64 = base64.b64encode(arr_bytes)
            arr_str = str(arr_b64, encoding="utf-8")
            return arr_str

        vertices_pos = arr_to_string(self.vertices, np.float32)
        triangles = arr_to_string(self.triangles, np.uint32)
        dct = {"vertices": vertices_pos, "triangles": triangles}

        return dct

    def apply_transformation(self, transf: TransformationsMatrix, invert_transf: bool = False):
        """Apply transformation in geometry"""

        self.vertices = transf.apply_points(self.vertices, invert_transf=invert_transf)
        self._full_update()

    def apply_transformation_matrix(self, M: np.ndarray, invert_transf: bool = False):
        """Apply transformation in geometry"""

        self.vertices = apply_transformation_matrix(
            self.vertices, M, arr_type="point", invert_transf=invert_transf
        )
        self._full_update()

    def binary_stl(self) -> bytes:
        """Get lagrangian geometry as STL binary format"""
        data = stl_binary(self.triangle_vertices, self.normals)
        return data

    def export_stl(self, filename: pathlib.Path):
        """Export lagrangian geometry in STL format

        Args:
            filename (pathlib.Path): filename to save to
        """

        data = self.binary_stl()

        filename.parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "wb") as f:
            f.write(data)

    def triangles_inside_volume(
        self, start: tuple[float, ...], end: tuple[float, ...]
    ) -> np.ndarray:
        """Get indexes of triangles that are inside volume

        Args:
            start (tuple[float, ...]): Volume start
            end (tuple[float, ...]): Volume end

        Returns:
            np.ndarray: indexes of triangles that are inside volume
        """

        start_arr = np.array(start, dtype=np.float32)
        end_arr = np.array(end, dtype=np.float32)

        # This simplification is good enough for our use case.
        # But vertices/triangles in volume borders may not give the desired result.
        bigger_start = (self.vertices >= start_arr).all(axis=1)
        smaller_end = (self.vertices <= end_arr).all(axis=1)
        bool_arr = np.logical_and(bigger_start, smaller_end)

        bool_triangles = np.zeros((self.triangles.shape[0],), dtype=bool)
        for t_idx, t in enumerate(self.triangles):
            if any(bool_arr[v_idx] for v_idx in t):
                bool_triangles[t_idx] = True

        return bool_triangles

    def join(self, geometries_list: list[LnasGeometry]):
        """Join into this geometry a list of LnasGeometry

        Args:
            geometries_list (list[LnasGeometry]): List of LnasGeometry to be combined
        """
        if len(geometries_list) < 1:
            raise ValueError(
                "No geometry to combine. It must be a list of at least two LnasGeometry"
            )

        for geometry in geometries_list:
            new_tri = geometry.triangles.copy() + len(self.vertices)
            self.vertices = np.vstack((self.vertices, geometry.vertices.copy()))
            self.triangles = np.vstack((self.triangles, new_tri))

        self._full_update()
