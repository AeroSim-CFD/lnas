# CHANGELOG

## 0.6.9

* `LnasFormat.from_stl` now matches the behaviour of the Rust `stl2lnas` command:
  * Degenerate triangles (area < 1e-5) are discarded on load
  * Vertices are deduplicated with 5-decimal-place precision
  * A surface entry keyed by the file stem is populated automatically
* Merged `feature/invalid-tri`: invalid normals during transformation now remove triangles instead of raising an error

## 0.6.8

Bugs:

* Fixed `correct_inverted_normals` applying the `p1/p2` swap to all triangles instead of only the ones with an inverted normal

## 0.6.7

Bugs:

* Corrected bug introduced by 0.6.6

## 0.6.6

* Changed behavior of export stl to return empty triangles instead of error when there is no triangle.