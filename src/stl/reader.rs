use crate::stl::triangle::TriangleSTL;
use crate::utils::{bytes_to_u32_le, Vec3f};
use std::{convert::TryInto, fs, path};

const TRIANGLE_BYTES_SIZE: usize = 50;
const HEADER_BYTES_SIZE: usize = 80;

fn read_file(filename: &path::PathBuf) -> Vec<u8> {
    let content = fs::read(filename).expect(format!("Unable to read {:?}", filename).as_str());
    return content;
}

pub fn read_stl(filename: &path::PathBuf) -> Vec<TriangleSTL> {
    let stl_content = read_file(filename);
    let n_triangles = number_of_triangles(&stl_content);
    let triangles = triangles_from_stl(&stl_content, n_triangles);
    return triangles;
}

fn number_of_triangles(stl_content: &Vec<u8>) -> u32 {
    // 80 bytes of header, then the number of triangles as u32 (4 bytes)
    let n_triangles_bytes: [u8; 4] = stl_content[HEADER_BYTES_SIZE..HEADER_BYTES_SIZE + 4]
        .try_into()
        .expect("Invalid file format (number of triangles)");
    let n_triangles = bytes_to_u32_le(&n_triangles_bytes);
    return n_triangles;
}

fn bytes_to_triangle(b: &[u8; TRIANGLE_BYTES_SIZE]) -> TriangleSTL {
    let normal = Vec3f::from_bytes_le(&b[..12].try_into().expect("Invalid file format (point 0)"));
    let point0 =
        Vec3f::from_bytes_le(&b[12..24].try_into().expect("Invalid file format (point 1)"));
    let point1 =
        Vec3f::from_bytes_le(&b[24..36].try_into().expect("Invalid file format (point 2)"));
    let point2 = Vec3f::from_bytes_le(&b[36..48].try_into().expect("Invalid file format (normal)"));
    // 2 points are for attribute byte count
    return TriangleSTL::new(point0, point1, point2, normal);
}

fn triangles_from_stl(stl_content: &Vec<u8>, n_triangles: u32) -> Vec<TriangleSTL> {
    // + 4 due to triangle numbers
    let start_byte = HEADER_BYTES_SIZE + 4;
    let mut all_triangles: Vec<TriangleSTL> = Vec::new();
    let mut n_invalid_triangles = 0;
    for i in 0..n_triangles {
        let curr_byte_idx = start_byte + TRIANGLE_BYTES_SIZE * (i as usize);
        let curr_triangle_bytes: &[u8; TRIANGLE_BYTES_SIZE] = stl_content
            [curr_byte_idx..curr_byte_idx + TRIANGLE_BYTES_SIZE]
            .try_into()
            .expect("Invalid triangle bytes");
        let triangle = bytes_to_triangle(curr_triangle_bytes);
        if triangle.check_area_valid() {
            all_triangles.push(triangle);
        } else {
            n_invalid_triangles += 1;
        }
    }
    if n_invalid_triangles > 0 {
        println!(
            "Found {} invalid triangles in STL, they were not added to LNAS",
            n_invalid_triangles
        );
    }

    return all_triangles;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_read_stl_cube() {
        let str_filename = String::from("examples/stl/cube.stl");
        let filename = path::Path::new(str_filename.as_str()).to_owned();
        let triangles = read_stl(&filename);
        // Cube has 2 triangles each face
        assert_eq!(triangles.len(), 6 * 2);
    }

    #[test]
    fn can_read_stl_terrain() {
        let str_filename = String::from("examples/stl/terrain.stl");
        let filename = path::Path::new(str_filename.as_str()).to_owned();
        read_stl(&filename);
    }
}
