# LNAS

Lagrangian Nassu (`.lnas`) is a geometry format to extend STL and minimize space used to represent geometries.

This repo contains the Python API (`lnas` folder) and the program for file conversion, `stl2lnas`, written in Rust.

## Dependencies

The main executable, `stl2lnas` is written on Rust. So in order to use it, you need to [install the Rust-Lang tools](https://www.rust-lang.org/tools/install).
After that, you may convert your STL files.

## `stl2lnas` installation

To run `stl2lnas` from anywhere on your system, run the commands below

```bash
cd stl2lnas
# Build release version
cargo build --release
# Copy the executable to your system's PATH
cp target/release/stl2lnas ~/.local/bin/
```

After that, `stl2lnas` is treated as a command in your system. You may run using

```bash
# --dir/-d <folder>: Folders with STL files
# --file/-f <file>: STL filenames
# -o: output to save .lnas file
# --overwrite: Add this if you wish to overwrite previously generated files
# --copy-stl: Add this if you wish to copy STL files to output
stl2lnas --dir examples/stl/folder_example \
  -d another/folder/with/stl \
  --file examples/stl/cube.stl \
  -f examples/stl/cylinder.stl \
  -o output/converted.lnas \
  --overwrite \
  --copy-stl
```

This outputs the file and, if `--copy-stl` is provided, a folder as `<output>.stls/` with the STLs used for generation and its names.

### Debug

To run the program for debug purposes, use

```bash
# --: finish cargo's arguments
# --dir/-d <folder>: Folders with STL files
# --file/-f <file>: STL filenames
# -o: output to save .lnas file
# --overwrite: Add this if you wish to overwrite previously generated files
cargo run -- \
  --dir examples/stl/folder_example \
  -d another/folder/with/stl \
  --file examples/stl/cube.stl \
  -f examples/stl/cylinder.stl \
  -o output/converted.lnas
```

## `lnas` API

To use `lnas` API, install the package using

```bash
pip install aerosim-lnas
```

## Lagrangian Nassu format (.lnas)

The Lagrangian Nassu format contains informations for representing a body. 
It follows similar compact strategy as [Wavefront obj format](https://en.wikipedia.org/wiki/Wavefront_.obj_file), restricting its polygons to triangles.

The format is used to define nodes (points) that are used by IBM (Immersed Boundary Method) to represent a body and its physics.

The format definition is:

```yaml
# Format version. Every major, ".lnas" breaks compatibility 
# v0.2.1 is not compatible with v0.1.0, but it is with v0.2.0
version: "v0.5.0"
geometry:
  # Vertices are represented as a list [(x0, y0, z0), (x1, y1, z1), ..., (xk, yk, zk)] in f32
  vertices: <base64>
  # Triangles are represented as a list [(v01, v02, v03), (v11, v12, v13), ..., (vn1, vn2, vn3)] in u32
  # Where each value in the triple is the point index in the `vertices` list.
  triangles: <base64>
  # Normal may be recovered using "right hand" rule, same convention as OpenGL.
  # That is, it considers rotation in sequence p1->p2->p3, so normal is
  # U = p2 - p1; V = p3 - p1 then the normal N = U X V
  # https://www.khronos.org/opengl/wiki/Calculating_a_Surface_Normal

# Surfaces are patches of triangles that describe a given set of triangles.
# It's used in post processing cases, when a geometry may be divided in multiple surfaces for
# post processing.
# The keys are the names of the .stl used
surfaces:
  # Surface name as key in dictionary
  surface1:
    # Index of triangles in given surface, referencing the triangles in `geometry.triangles`
    # It's represented as [t_idx1, t_idx2, t_idx3, ...] in u32
    triangles_idxs: <base64>
  # Other surfaces...
  surface2:
    triangles_idxs: <base64>
```

### Compactation impact

The compactation of `.lnas` format is mainly due to not repeating the vertices shared between triangles.
So the impact of it increases with the number of shared vertices.

It also uses the points order in the triangle to store the triangle's normal information, not needing to explicitly specify it.

The relative size from STL to LNAS may be calculated by:

$V_{LNAS} = 3 T_{STL}/S$

$T_{LNAS} = 0.5 T_{STL}/3$

Where $V_{LNAS}$ is the number of unique vertices, $T_{LNAS}$ the number of indexes to represent triangles, $T_{STL}$ the number of triangles and $S$ the average number of triangles that a vertice shares.

The impact of not saving normal is showed as the 0.5 in $T_{LNAS}$ and saving the points as index is the /3 in the same variable.
While storing only once shared vertices lowers the $V_{LNAS}$ by the $S$ factor.

The total size to represent STL is $3 \cdot 4T_{STL}$ (3 vertices and one normal for each triangle, with 3 dimensions each).

The size to represent STL in LNAS format is $3V_{LNAS}+3T_{LNAS}$.

For example, for a STL that has 100.000 triangles, where each vertex is shared in average among 4 triangles.
There are $3*120.000/4=90.000$ unique vertices and $0.5 \cdot 130.000/3 = 20.000$.

So $3V_{LNAS}+3T_{LNAS}=270.000+60.000=330.000$ and $12 \cdot 100.000=120.0000$.
**The relative size of LNAS format is $330.000/1.200.000=27.5$% in this case**.

## Limitations

Some of the known limitations are:

- It can only convert binary STL files
