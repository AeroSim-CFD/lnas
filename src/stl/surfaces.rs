use std::{collections::HashMap, path, string::String};

use crate::stl::reader::read_stl;
use crate::stl::triangle::TriangleSTL;

fn get_stl_triangles(stl_filename: &path::PathBuf) -> Vec<TriangleSTL> {
    let triangles = read_stl(stl_filename);
    return triangles;
}

pub fn get_surfaces(
    files: &HashMap<String, path::PathBuf>,
) -> (Vec<TriangleSTL>, HashMap<String, Vec<u32>>) {
    let mut all_triangles: Vec<TriangleSTL> = Vec::new();
    let mut surfaces_triangles: HashMap<String, Vec<u32>> = HashMap::new();

    // Do this to be deterministic
    let mut surface_names: Vec<&String> = files.keys().into_iter().collect();
    surface_names.sort();

    for surface_name in surface_names.into_iter() {
        // STL triangles
        let stl_filename = files.get(surface_name).unwrap();
        let mut stl_triangles = get_stl_triangles(stl_filename);
        // Index of these STL triangles when comparing to list of triangles
        let triangles_idxs_range = all_triangles.len()..all_triangles.len() + stl_triangles.len();
        let triangles_idxs_u32: Vec<u32> = triangles_idxs_range
            .map(|v| u32::try_from(v).unwrap())
            .collect();

        // Add STL triangles to all triangles
        all_triangles.append(&mut stl_triangles);
        // Inset surface triangles indexes in hash map
        surfaces_triangles.insert(surface_name.to_owned(), triangles_idxs_u32);
    }
    return (all_triangles, surfaces_triangles);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_read_stl_cube() {
        let mut files: HashMap<String, path::PathBuf> = HashMap::new();
        files.insert(
            "cube".to_string(),
            path::Path::new("examples/stl/cube.stl".to_string().as_str()).to_owned(),
        );
        let (triangles, surfaces) = get_surfaces(&files);
        // Cube has 2 triangles each face
        let surface_cube = surfaces.get("cube").unwrap().to_owned();
        assert_eq!(triangles.len(), surface_cube.len());
        assert_eq!(
            surface_cube,
            (0..surface_cube.len())
                .map(|v| u32::try_from(v).unwrap())
                .collect::<Vec<u32>>()
        );
    }

    #[test]
    fn can_read_stl_combine() {
        let mut files: HashMap<String, path::PathBuf> = HashMap::new();
        files.insert(
            "cube".to_string(),
            path::Path::new("examples/stl/cube.stl".to_string().as_str()).to_owned(),
        );
        files.insert(
            "cylinder".to_string(),
            path::Path::new("examples/stl/cylinder.stl".to_string().as_str()).to_owned(),
        );

        let (triangles, surfaces) = get_surfaces(&files);
        let surface_cube = surfaces.get("cube").unwrap().to_owned();
        let surface_cylinder = surfaces.get("cylinder").unwrap().to_owned();

        assert_eq!(triangles.len(), surface_cube.len() + surface_cylinder.len());
        // Values are sorted by surface name
        assert_eq!(
            surface_cube,
            (0..surface_cube.len())
                .map(|v| u32::try_from(v).unwrap())
                .collect::<Vec<u32>>()
        );
        assert_eq!(
            surface_cylinder,
            (surface_cube.len()..surface_cube.len() + surface_cylinder.len())
                .map(|v| u32::try_from(v).unwrap())
                .collect::<Vec<u32>>()
        );
    }
}
