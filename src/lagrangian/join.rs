use crate::lagrangian::triangle::LagrangianTriangle;
use crate::lagrangian::vertice::LagrangianVertice;
use crate::utils::{Vec3f, Vec3u};
use std::collections::HashMap;

fn get_vertices_vector(lagrangian_vertices: &HashMap<LagrangianVertice, usize>) -> Vec<Vec3f> {
    let mut vec_vertices: Vec<Vec3f> = Vec::with_capacity(lagrangian_vertices.len());
    for _ in lagrangian_vertices.iter() {
        vec_vertices.push(Vec3f {
            x: 0f32,
            y: 0f32,
            z: 0f32,
        });
    }
    for (vert, idx) in lagrangian_vertices.iter() {
        vec_vertices[idx.clone()] = vert.pos.clone();
    }
    return vec_vertices;
}

fn get_triangles_vector(
    lagrangian_vertices: &HashMap<LagrangianVertice, usize>,
    lagrangian_triangles: &Vec<LagrangianTriangle>,
) -> Vec<Vec3u> {
    let mut vec_triangles: Vec<Vec3u> = Vec::with_capacity(lagrangian_triangles.len());
    for t in lagrangian_triangles.iter() {
        let idx0 = lagrangian_vertices.get(&t.p0).unwrap().clone();
        let idx1 = lagrangian_vertices.get(&t.p1).unwrap().clone();
        let idx2 = lagrangian_vertices.get(&t.p2).unwrap().clone();
        let vec_u = Vec3u {
            x: idx0 as u32,
            y: idx1 as u32,
            z: idx2 as u32,
        };
        vec_triangles.push(vec_u);
    }
    return vec_triangles;
}

pub fn join_information(
    lagrangian_vertices: &HashMap<LagrangianVertice, usize>,
    lagrangian_triangles: &Vec<LagrangianTriangle>,
) -> (Vec<Vec3f>, Vec<Vec3u>) {
    let vec_vertices = get_vertices_vector(lagrangian_vertices);
    let vec_triangles = get_triangles_vector(lagrangian_vertices, lagrangian_triangles);

    return (vec_vertices, vec_triangles);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lagrangian::triangle::generate_lagrangian_triangles;
    use crate::lagrangian::vertice::generate_lagrangian_vertices;
    use crate::stl::reader::read_stl;
    use std::path;

    fn check_vertices_compatibility(
        lagrangian_vertices: &HashMap<LagrangianVertice, usize>,
        joined_vertices: &Vec<Vec3f>,
    ) {
        assert_eq!(lagrangian_vertices.len(), joined_vertices.len());
        for (i, vert) in joined_vertices.iter().enumerate() {
            let lagr_vert = LagrangianVertice::new(vert.clone());
            let lagr_vert_pos = lagrangian_vertices.get(&lagr_vert).unwrap().clone();
            assert_eq!(i as usize, lagr_vert_pos)
        }
    }

    fn check_triangles_compatibility(
        lagrangian_triangles: &Vec<LagrangianTriangle>,
        joined_vertices: &Vec<Vec3f>,
        joined_triangles: &Vec<Vec3u>,
    ) {
        assert_eq!(joined_triangles.len(), lagrangian_triangles.len());
        for (i, triangle) in joined_triangles.iter().enumerate() {
            let lagr_triangle = lagrangian_triangles[i];

            let p0 = lagr_triangle.p0;
            let p0_idx = triangle.x;
            let p0_lagr = joined_vertices[p0_idx as usize];

            if p0.pos != p0_lagr {
                panic!("P0 is wrong");
            }

            let p1 = lagr_triangle.p1;
            let p1_idx = triangle.y;
            let p1_lagr = joined_vertices[p1_idx as usize];

            if p1.pos != p1_lagr {
                panic!("P1 is wrong");
            }

            let p2 = lagr_triangle.p2;
            let p2_idx = triangle.z;
            let p2_lagr = joined_vertices[p2_idx as usize];

            if p2.pos != p2_lagr {
                panic!("P2 is wrong");
            }
        }
    }

    #[test]
    fn check_join_info_stl_cube() {
        let filename = path::Path::new(String::from("examples/stl/cube.stl").as_str()).to_owned();
        let triangles = read_stl(&filename);
        let lagr_vertices = generate_lagrangian_vertices(&triangles);
        let lagr_triangles = generate_lagrangian_triangles(&lagr_vertices, &triangles);
        let (joined_vert, joined_tri) = join_information(&lagr_vertices, &lagr_triangles);

        check_vertices_compatibility(&lagr_vertices, &joined_vert);
        check_triangles_compatibility(&lagr_triangles, &joined_vert, &joined_tri);
    }

    #[test]
    fn check_join_infocar_stl_terrain() {
        let filename =
            path::Path::new(String::from("examples/stl/terrain.stl").as_str()).to_owned();
        let triangles = read_stl(&filename);
        let lagr_vertices = generate_lagrangian_vertices(&triangles);
        let lagr_triangles = generate_lagrangian_triangles(&lagr_vertices, &triangles);
        let (joined_vert, joined_tri) = join_information(&lagr_vertices, &lagr_triangles);

        check_vertices_compatibility(&lagr_vertices, &joined_vert);
        check_triangles_compatibility(&lagr_triangles, &joined_vert, &joined_tri);
    }
}
