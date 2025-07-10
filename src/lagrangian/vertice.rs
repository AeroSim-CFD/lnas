use crate::stl::triangle::TriangleSTL;
use crate::utils::Vec3f;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Lagranngian node is defined by a position, with normal and area properties
#[derive(Serialize, Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub struct LagrangianVertice {
    pub pos: Vec3f,
}

impl LagrangianVertice {
    pub fn new(pos: Vec3f) -> LagrangianVertice {
        return LagrangianVertice { pos };
    }

    pub fn get_le_bytes(&self) -> Vec<u8> {
        let vec = self.pos.to_le_bytes_as_f32();

        return vec.to_vec();
    }
}

impl fmt::Display for LagrangianVertice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return write!(f, "pos {}", self.pos);
    }
}

/// Convert STL triangles to lagranagian nodes
pub fn generate_lagrangian_vertices(
    triangles: &Vec<TriangleSTL>,
) -> HashMap<LagrangianVertice, usize> {
    let mut hash_vertices: HashSet<&Vec3f> = HashSet::new();

    for t in triangles.iter() {
        for p in [&t.point0, &t.point1, &t.point2] {
            hash_vertices.insert(p);
        }
    }
    let ordered_vertices: Vec<&Vec3f> = hash_vertices.into_iter().collect();
    // ordered_vertices.sort();

    let mut lagrangian_vertices: HashMap<LagrangianVertice, usize> = HashMap::new();
    for (i, p) in ordered_vertices.into_iter().enumerate() {
        let k = LagrangianVertice::new(*p);
        lagrangian_vertices.insert(k, i);
    }
    return lagrangian_vertices;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stl::reader::read_stl;
    use std::path;

    #[test]
    fn check_vertices_stl_cube() {
        let filename = path::Path::new(String::from("examples/stl/cube.stl").as_str()).to_owned();
        let triangles = read_stl(&filename);
        // Cube has 2 triangles each face
        assert_eq!(triangles.len(), 6 * 2);
        let lagr_vertices = generate_lagrangian_vertices(&triangles);
        // Cube has 8 vertices
        assert_eq!(lagr_vertices.len(), 8);
    }

    #[test]
    fn check_vertices_stl_terrain() {
        let filename =
            path::Path::new(String::from("examples/stl/terrain.stl").as_str()).to_owned();
        let triangles = read_stl(&filename);
        let lagr_vertices = generate_lagrangian_vertices(&triangles);
        if triangles.len() * 3 < lagr_vertices.len() {
            panic!("There are more vertices than 3 times triangles");
        }
        println!(
            "Triangles {} lagr vertices {} avrg shared {:.4}",
            triangles.len(),
            lagr_vertices.len(),
            3f32 * (triangles.len() as f32) / (lagr_vertices.len() as f32)
        );
    }
}
