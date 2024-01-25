import numpy as np
from lnas import LnasFormat, LnasGeometry


def filter_from_list(surface_list: list[str], fmt: LnasFormat) -> tuple[LnasGeometry, np.ndarray]:
    """Filters the mesh from a list of surfaces

    Args:
        surface_list (list[str]): List of surfaces to be filtered
        fmt (LnasFormat): LNAS format with every surface available

    Returns:
        tuple[LnasGeometry, np.ndarray]: Tuple with filtered LNAS mesh geometry
        and the filtered triangle indices from the input format
    """
    if len(surface_list) == 0:
        raise Exception("No geometry could be filtered from the list of surfaces.")

    geom_mesh = LnasGeometry(
        vertices=fmt.geometry.vertices, triangles=np.empty((0, 3), dtype=np.uint32)
    )
    geom_triangles_idxs = np.array([], dtype=np.uint32)

    for sfc in surface_list:
        m = fmt.geometry_from_surface(sfc)
        geom_mesh.triangles = np.vstack((geom_mesh.triangles, m.triangles))
        geom_triangles_idxs = np.hstack((geom_triangles_idxs, fmt.surfaces[sfc].copy()))

    geom_mesh._full_update()

    return geom_mesh, geom_triangles_idxs


def combine_geometries(geometries_list: list[LnasGeometry]) -> LnasGeometry:
    """Combine a list of LnasGeometry into a single LnasGeometry

    Args:
        geometries_list (list[LnasGeometry]): List of LnasGeometry to be combined

    Returns:
        LnasGeometry: Result of the combination of a list of LnasGeometry
    """
    if len(geometries_list) < 2:
        raise ValueError("No geometry to combine. It must be a list of at least two LnasGeometry")

    result_geometry = LnasGeometry(
        vertices=np.empty((0, 3), dtype=np.uint32), triangles=np.empty((0, 3), dtype=np.uint32)
    )

    for geometry in geometries_list:
        new_tri = geometry.triangles.copy() + len(result_geometry.vertices)
        result_geometry.vertices = np.vstack((result_geometry.vertices, geometry.vertices.copy()))
        result_geometry.triangles = np.vstack((new_tri, geometry.triangles.copy()))

    result_geometry._full_update()

    return result_geometry


def combine_lnas(
    lnas_fmts: list[LnasFormat], surfaces_suffixes: list[str] | None = None
) -> LnasFormat:
    """Combines a list of LnasFormat into a new one. The indexing follows the list sequence

    Args:
        lnas_fmts (list[LnasFormat]): List of LnasFormat to be combined
        surfaces_suffixes (list[str] | None, optional): Optional suffix list to add to each lnas. Defaults to None.

    Raises:
        ValueError: There is no LnasFormat in the list
        ValueError: The size of the list of suffixes is smaller than the format list
        KeyError: If any surface name repeats, then a suffix must be added to one of the surface's LnasFormat

    Returns:
        LnasFormat: Combined LnasFormat
    """
    if len(lnas_fmts) == 0:
        raise ValueError("No LNAS to combine")
    if surfaces_suffixes is not None:
        if len(surfaces_suffixes) < len(lnas_fmts):
            raise ValueError("Less surfaces suffixes than required")
    merged_geo = lnas_fmts[0].geometry.copy()
    surfaces = {}

    suffix = surfaces_suffixes[0] if surfaces_suffixes is not None else ""
    for s, arr in lnas_fmts[0].surfaces.items():
        surfaces[s + suffix] = arr.copy()

    for i, lnas_fmt in enumerate(lnas_fmts):
        if i == 0:
            # Already added
            continue
        n_verts, n_tris = len(merged_geo.vertices), len(merged_geo.triangles)

        verts_add = lnas_fmt.geometry.vertices.copy()
        merged_geo.vertices = np.concatenate((merged_geo.vertices.copy(), verts_add), axis=0)

        tri_add = lnas_fmt.geometry.triangles + n_verts
        merged_geo.triangles = np.concatenate((merged_geo.triangles.copy(), tri_add), axis=0)

        suffix = surfaces_suffixes[i] if surfaces_suffixes is not None else ""
        for s, arr in lnas_fmt.surfaces.items():
            key = s + suffix
            if key in surfaces:
                raise KeyError(
                    f"Surface {s} is already in the list of surfaces, provide a suffix for it"
                )
            surfaces[key] = arr + n_tris

    merged_geo._full_update()

    return LnasFormat(version=lnas_fmts[0].version, geometry=merged_geo, surfaces=surfaces)
