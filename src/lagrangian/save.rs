use crate::lagrangian::format::LNAS;
use crate::utils::create_folder_for_filename;
use std::error::Error;
use std::{fs, path};

use serde_yaml;

pub fn save_lnas(filename: &path::Path, lnas_obj: &LNAS) -> Result<(), Box<dyn Error>> {
    create_folder_for_filename(filename)?;
    let file = fs::File::create(filename)?;
    println!("Saving...");
    serde_yaml::to_writer(file, &lnas_obj)?;

    return Ok(());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lagrangian::format::get_lnas_obj_save;
    use crate::lagrangian::join::join_information;
    use crate::lagrangian::triangle::generate_lagrangian_triangles;
    use crate::lagrangian::vertice::generate_lagrangian_vertices;
    use crate::stl::surfaces::get_surfaces;
    use std::{collections::HashMap, path, string::String};

    #[test]
    fn check_join_info_stl_cube() {
        let mut files: HashMap<String, path::PathBuf> = HashMap::new();
        let filename: String = "examples/stl/cube.stl".to_string();
        files.insert(
            "cube".to_string(),
            path::Path::new(filename.as_str()).to_owned(),
        );

        let (triangles, surfaces) = get_surfaces(&files);
        let lagr_vertices = generate_lagrangian_vertices(&triangles);
        let lagr_triangles = generate_lagrangian_triangles(&lagr_vertices, &triangles);
        let (joined_vertices, joined_triangles) = join_information(&lagr_vertices, &lagr_triangles);
        let lnas_obj = get_lnas_obj_save(&joined_vertices, &joined_triangles, &surfaces);

        let folder_path = path::Path::new("output/");
        let lnas_filename = folder_path.join(format!("{}.lnas", "cube"));

        save_lnas(&lnas_filename, &lnas_obj).unwrap();
    }

    #[test]
    fn check_join_info_stl_terrain() {
        // let filename_cfg = String::from("examples/convert_terrain.yaml");
        let mut files: HashMap<String, path::PathBuf> = HashMap::new();
        let filename: String = "examples/stl/terrain.stl".to_string();
        files.insert(
            "cube".to_string(),
            path::Path::new(filename.as_str()).to_owned(),
        );

        let (triangles, surfaces) = get_surfaces(&files);
        let lagr_vertices = generate_lagrangian_vertices(&triangles);
        let lagr_triangles = generate_lagrangian_triangles(&lagr_vertices, &triangles);
        let (joined_vertices, joined_triangles) = join_information(&lagr_vertices, &lagr_triangles);
        let lnas_obj = get_lnas_obj_save(&joined_vertices, &joined_triangles, &surfaces);

        let folder_path = path::Path::new("output/");
        let lnas_filename = folder_path.join(format!("{}.lnas", "terrain"));

        save_lnas(&lnas_filename, &lnas_obj).unwrap();
    }

    #[test]
    fn check_join_info_stl_terrain_cube() {
        let mut files: HashMap<String, path::PathBuf> = HashMap::new();
        let filename: String = "examples/stl/plane.stl".to_string();
        files.insert(
            "plane".to_string(),
            path::Path::new(filename.as_str()).to_owned(),
        );
        let filename: String = "examples/stl/cube.stl".to_string();
        files.insert(
            "cube".to_string(),
            path::Path::new(filename.as_str()).to_owned(),
        );

        let (triangles, surfaces) = get_surfaces(&files);
        let lagr_vertices = generate_lagrangian_vertices(&triangles);
        let lagr_triangles = generate_lagrangian_triangles(&lagr_vertices, &triangles);
        let (joined_vertices, joined_triangles) = join_information(&lagr_vertices, &lagr_triangles);
        let lnas_obj = get_lnas_obj_save(&joined_vertices, &joined_triangles, &surfaces);

        let folder_path = path::Path::new("output/");
        let lnas_filename = folder_path.join(format!("{}.lnas", "plane_cube"));

        save_lnas(&lnas_filename, &lnas_obj).unwrap();
    }
}
