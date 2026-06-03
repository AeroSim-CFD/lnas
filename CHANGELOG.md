# CHANGELOG

## Unreleased

* Vectorized `LnasGeometry._update_vertices_normals` with `np.add.at`, removing the per-triangle Python loop that was `O(n_triangles)` interpreted and slow on large meshes (#18)
* Vectorized `LnasGeometry.triangles_inside_volume`, replacing the per-triangle Python loop with a single masked index lookup
* `LnasGeometry._full_update` now computes the triangle cross product (and its norm) once and shares it between normals and areas, instead of recomputing it twice per update on the geometry hot path
* `LnasFormat.geometry_from_list_surfaces` no longer runs a redundant second `_full_update`, and `filter_triangles` no longer copies the full triangle array only to discard it
* Vectorized the triangle-index offset computation in `LnasFormat.filter_triangles`, replacing the per-triangle Python loop with a cumulative sum

Bugs:

* Fixed `LnasGeometry.__eq__` and `LnasFormat.__eq__` raising `AttributeError` when compared against a non-geometry/non-format object; they now return `False`
* `LnasVersionError` now subclasses `Exception` instead of `BaseException`, so it is caught by normal `except Exception` handlers
* `apply_transformation_matrix` now raises a clear `ValueError` for an unknown `arr_type` instead of failing later with an unbound-variable `NameError`

## 0.6.9

* `LnasFormat.from_stl` now matches the behaviour of the Rust `stl2lnas` command:
  * Degenerate triangles (area < 1e-5) are discarded on load
  * Vertices are deduplicated with 5-decimal-place precision
  * A surface entry keyed by the file stem is populated automatically
* Merged `feature/invalid-tri`: invalid normals during transformation now remove triangles instead of raising an error
* Added CI tests comparing Python `from_stl` output against the Rust `stl2lnas` binary

## 0.6.8

Bugs:

* Fixed `correct_inverted_normals` applying the `p1/p2` swap to all triangles instead of only the ones with an inverted normal

## 0.6.7

Bugs:

* Corrected bug introduced by 0.6.6

## 0.6.6

* Changed behavior of export stl to return empty triangles instead of error when there is no triangle.