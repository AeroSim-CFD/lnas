# RELEASE NOTES (`stl2lnas`)

## v0.5.2

- Migrated Git
- Updated dependencies

## v0.5.1

- Changed behavior of STL copy, now only copies to output if `--copy-stl` is provided
- Now file version is automatically get from the pacakge metadata

## v0.5.0

- Removed support for normalization
- Removed unused bin script (`convert_lnas_folder`)
- Updated dependencies (clap, serde and base64)
- Changed usage for CLI instead of configuration file
- Updated README

------------------------------------------------------------------------------

## v0.4.3

- Added support for loading STL from folder

## v0.4.2

- Added check for small triangles to not add then

## v0.4.1

- Added bin to convert lnas in folder

## v0.4.0

- Added support to generate .lnas from multiple STLs
- Added surfaces in .lnas format
  - Each STL used becomes a surface in the output file
- Set normalization as an optional parameter for configuration

------------------------------------------------------------------------------

## v0.3.0

- Added normalization direction as options
- Changed fields of .lnas format and conversion file

------------------------------------------------------------------------------

## v0.2.1

- Corrected bug in normal direction, now doing orientation check

------------------------------------------------------------------------------

## v0.2.0

- Updated documentation
- Updated .lnas file format to reduce space used
- Transfered most division responsabilities to reader

Code:

- Added tests to modules
- Refactored file organization

------------------------------------------------------------------------------