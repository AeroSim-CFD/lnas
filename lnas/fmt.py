from __future__ import annotations

import base64
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
from lnas import LnasGeometry
from lnas.exceptions import LnasVersionError
from lnas.utils import read_yaml, save_yaml

_CURR_MAJOR_VERSION = "v0.4"


@dataclass
class LagrangianNormalization:
    size: float
    direction: str


@dataclass
class LnasFormat:
    """Lagrangian format description"""

    version: str
    name: str
    normalization: Optional[LagrangianNormalization]
    geometry: LnasGeometry
    surfaces: dict[str, np.ndarray]

    def __eq__(self, __o: object) -> bool:
        if type(self) != type(__o):
            return False

        if not (
            self.version == __o.version
            and self.name == __o.name
            and self.normalization == __o.normalization
            and self.geometry == __o.geometry
            and set(self.surfaces.keys()) == set(__o.surfaces.keys())
        ):
            return False
        for k in self.surfaces.keys():
            if not np.array_equal(self.surfaces[k], __o.surfaces[k]):
                return False
        return True

    def geometry_from_surface(self, surface_name: str) -> LnasGeometry:
        if surface_name not in self.surfaces:
            raise KeyError(
                f"Unable to find surface named {surface_name}. "
                + f"Available ones are {list(self.surfaces.keys())}"
            )

        triangles_idxs = self.surfaces[surface_name].copy()
        triangles = self.geometry.triangles[triangles_idxs].copy()
        vertices = self.geometry.vertices.copy()
        return LnasGeometry(vertices=vertices, triangles=triangles)

    @classmethod
    def from_dct(cls, dct: dict[str, Any]) -> LnasFormat:
        """Load lagrangian format from dictionary"""

        version = str(dct["version"])
        if version[:-2] != _CURR_MAJOR_VERSION:
            raise LnasVersionError(
                f"LNAS version {version} is uncompatible with reader version {_CURR_MAJOR_VERSION}"
            )
        name = str(dct["name"])
        normalization = (
            LagrangianNormalization(**dct["normalization"])
            if dct["normalization"] is not None
            else None
        )
        geometry = LnasGeometry.from_dct(dct["geometry"])
        surfaces: dict[str, np.ndarray] = {}
        for surface_name, surface_b64 in dct["surfaces"].items():
            surface_bytes = base64.b64decode(surface_b64)
            surface_arr = np.frombuffer(surface_bytes, dtype=np.uint32)
            surfaces[surface_name] = surface_arr

        return LnasFormat(
            version=version,
            name=name,
            normalization=normalization,
            geometry=geometry,
            surfaces=surfaces,
        )

    def to_dct(self) -> dict[str, Any]:
        """Get lagrangian format as dictionary"""

        dct = {}
        dct["version"] = str(self.version)
        dct["name"] = str(self.name)
        dct["normalization"] = (
            asdict(self.normalization) if self.normalization is not None else None
        )
        dct["geometry"] = self.geometry.to_dct()
        dct["surfaces"] = {}
        for surface_name, surface_arr in self.surfaces.items():
            surface_bytes = surface_arr.tobytes(order="C")
            surface_b64 = base64.b64encode(surface_bytes)
            dct["surfaces"][surface_name] = str(surface_b64, encoding="utf-8")

        return dct

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> LnasFormat:
        """Load lagrangian format from file"""

        try:
            dct_lnas = read_yaml(filename)
            return cls.from_dct(dct_lnas)
        except Exception as e:
            raise ValueError(f"Unable to read LNAS file {filename}") from e

    @classmethod
    def from_folder(cls, foldername: pathlib.Path, transformed: bool = False) -> LnasFormat:
        """Load lagrangian format from folder"""

        filename_cfg = foldername / "cfg.yaml"
        try:
            dct_cfg = read_yaml(filename_cfg)
            name = dct_cfg["name"]
        except Exception as e:
            raise FileNotFoundError(f"Unable to get LNAS name for folder {foldername}") from e
        if transformed:
            name += ".transformed"
        filename = foldername / (name + ".lnas")
        return cls.from_file(filename)

    def to_file(self, filename: pathlib.Path):
        """Save lagrangian format to file"""

        dct = self.to_dct()
        save_yaml(dct, filename)

    def export_stl(self, filename: pathlib.Path):
        """Export lagrangian geometry in STL format

        Args:
            filename (pathlib.Path): filename to save to
        """

        self.geometry.export_stl(filename)

    def filter_triangles(self, triangles_use: np.ndarray) -> LnasFormat:
        """Filter triangles of LNAS

        Args:
            triangles_use (np.ndarray): bool array of triangles to use

        Returns:
            LnasFormat: New LNAS with surfaces and geometry filtered
        """

        if len(triangles_use) != self.geometry.triangles.shape[0]:
            raise ValueError(
                "Invalid number of triangles to filter. "
                + f"{len(triangles_use)} != {self.geometry.triangles.shape[0]}"
            )

        geometry = self.geometry
        filtered_triangles = geometry.triangles[triangles_use].copy()

        new_geometry = self.geometry.copy()
        new_geometry.triangles = filtered_triangles
        new_geometry._full_update()

        # Build offset of filtered triangles considering past ones
        offset = 0
        offsets_arr = np.zeros((triangles_use.shape[0],), dtype=np.int32)
        for idx, v in enumerate(triangles_use):
            if not v:
                offset += 1
            else:
                offsets_arr[idx] = offset

        # Filter surfaces
        new_surfaces = {}
        for s, arr in self.surfaces.items():
            filtered_arr = np.extract(triangles_use[arr], arr).astype(np.int32)
            filtered_offset_arr = offsets_arr[filtered_arr].astype(np.int32)
            filtered_arr -= filtered_offset_arr
            new_surfaces[s] = filtered_arr

        new_lnas = LnasFormat(
            version=self.version,
            name=self.name,
            normalization=self.normalization,
            geometry=new_geometry,
            surfaces=new_surfaces,
        )

        return new_lnas
