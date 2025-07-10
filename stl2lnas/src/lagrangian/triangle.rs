use crate::lagrangian::vertice::LagrangianVertice;
use crate::stl::triangle::TriangleSTL;
use crate::utils::Vec3f;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt;

#[derive(Serialize, Clone, Copy, Hash, PartialEq, Eq)]
pub struct LagrangianTriangle {
    pub p0: LagrangianVertice,
    pub p1: LagrangianVertice,
    pub p2: LagrangianVertice,
}

fn get_normal_order(p0: Vec3f, p1: Vec3f, p2: Vec3f, normal: &Vec3f) -> (Vec3f, Vec3f, Vec3f) {
    for (pp0, pp1, pp2) in [(p0, p1, p2), (p0, p2, p1)] {
        // Same convention as OpenGL
        // https://www.khronos.org/opengl/wiki/Calculating_a_Surface_Normal
        // U = p1 - p0; V = p2 - p0
        let u = pp1 - pp0;
        let v = pp2 - pp0;
        let mut order_normal = u.cross(v);
        order_normal.normalize();

        // If same direction, dot product is equal
        if order_normal.dot(*normal) > 0f32 {
            return (pp0, pp1, pp2);
        }
    }

    panic!(
        "Normal invalid for triangle [{}, {}, {}] normal {}",
        p0, p1, p2, normal
    );
}

impl LagrangianTriangle {
    pub fn new(
        p0: LagrangianVertice,
        p1: LagrangianVertice,
        p2: LagrangianVertice,
        normal: &Vec3f,
    ) -> LagrangianTriangle {
        let (pn0, pn1, pn2) = get_normal_order(p0.pos, p1.pos, p2.pos, normal);
        return LagrangianTriangle {
            p0: LagrangianVertice::new(pn0),
            p1: LagrangianVertice::new(pn1),
            p2: LagrangianVertice::new(pn2),
        };
    }

    pub fn get_indexes(&self, triangles: &HashMap<LagrangianVertice, usize>) -> Vec<usize> {
        let idx0 = triangles.get(&self.p0).unwrap().clone();
        let idx1 = triangles.get(&self.p1).unwrap().clone();
        let idx2 = triangles.get(&self.p2).unwrap().clone();
        return [idx0, idx1, idx2].to_vec();
    }

    pub fn get_le_bytes(&self, triangles: &HashMap<LagrangianVertice, usize>) -> Vec<u8> {
        let vec = self.get_indexes(triangles);
        let vec_byte: Vec<u8> = vec.iter().flat_map(|x| x.to_le_bytes()).collect();
        return vec_byte;
    }
}

impl fmt::Display for LagrangianTriangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return write!(f, "p0 {} p1 {} p2 {}", self.p0, self.p1, self.p2);
    }
}

/// Convert STL triangles to lagranagian nodes
pub fn generate_lagrangian_triangles(
    _vertices: &HashMap<LagrangianVertice, usize>,
    triangles: &Vec<TriangleSTL>,
) -> Vec<LagrangianTriangle> {
    let mut lagrangian_triangles: Vec<LagrangianTriangle> = Vec::new();
    for t in triangles.iter() {
        let lagr_p0 = LagrangianVertice::new(t.point0.clone());
        let lagr_p1 = LagrangianVertice::new(t.point1.clone());
        let lagr_p2 = LagrangianVertice::new(t.point2.clone());
        let lagr_tri = LagrangianTriangle::new(lagr_p0, lagr_p1, lagr_p2, &t.normal);
        lagrangian_triangles.push(lagr_tri);
    }
    return lagrangian_triangles;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lagrangian::vertice::generate_lagrangian_vertices;
    use crate::stl::reader::read_stl;
    use std::path;

    #[test]
    fn check_normal_decision() {
        let p0 = Vec3f {
            x: 0f32,
            y: 0f32,
            z: 0f32,
        };
        let p1 = Vec3f {
            x: 1f32,
            y: 0f32,
            z: 0f32,
        };
        let p2 = Vec3f {
            x: 0f32,
            y: 1f32,
            z: 0f32,
        };
        let normal_pos = Vec3f {
            x: 0f32,
            y: 0f32,
            z: 1f32,
        };
        let normal_neg = Vec3f {
            x: 0f32,
            y: 0f32,
            z: -1f32,
        };
        let normal_pos_wrong = Vec3f {
            x: 0f32,
            y: 0f32,
            z: 0.5f32,
        };

        let res = get_normal_order(p0, p1, p2, &normal_pos);
        assert_eq!(res, (p0, p1, p2));

        let res = get_normal_order(p0, p1, p2, &normal_neg);
        assert_eq!(res, (p0, p2, p1));

        // Does not panic, gets normal in direction that dot product is positive
        let res = get_normal_order(p0, p1, p2, &normal_pos_wrong);
        assert_eq!(res, (p0, p1, p2));
    }

    #[test]
    fn check_triangles_stl_cube() {
        let filename = path::Path::new(String::from("examples/stl/cube.stl").as_str()).to_owned();
        let triangles = read_stl(&filename);
        let lagr_vertices = generate_lagrangian_vertices(&triangles);
        let lagr_triangles = generate_lagrangian_triangles(&lagr_vertices, &triangles);

        assert_eq!(lagr_triangles.len(), triangles.len());
    }

    #[test]
    fn check_triangles_stl_terrain() {
        let filename =
            path::Path::new(String::from("examples/stl/terrain.stl").as_str()).to_owned();
        let triangles = read_stl(&filename);
        let lagr_vertices = generate_lagrangian_vertices(&triangles);
        let lagr_triangles = generate_lagrangian_triangles(&lagr_vertices, &triangles);

        assert_eq!(lagr_triangles.len(), triangles.len());
    }
}
