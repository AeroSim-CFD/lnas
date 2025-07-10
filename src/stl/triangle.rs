use crate::assert_almost_equal;
use crate::utils;
use std::fmt;

#[derive(Clone, Copy, Hash)]
pub struct TriangleSTL {
    pub point0: utils::Vec3f,
    pub point1: utils::Vec3f,
    pub point2: utils::Vec3f,
    pub normal: utils::Vec3f,
}

impl TriangleSTL {
    pub fn new(
        point0: utils::Vec3f,
        point1: utils::Vec3f,
        point2: utils::Vec3f,
        normal: utils::Vec3f,
    ) -> TriangleSTL {
        assert_almost_equal!(&normal.norm(), 1.0, 1e-5f32);

        return TriangleSTL {
            point0,
            point1,
            point2,
            normal,
        };
    }

    pub fn normalize(&mut self, factor: f32, offset: utils::Vec3f) {
        self.point0.transform(factor, offset);
        self.point1.transform(factor, offset);
        self.point2.transform(factor, offset);
    }

    pub fn check_area_valid(self) -> bool {
        let u: utils::Vec3f = self.point0 - self.point1;
        let v: utils::Vec3f = self.point0 - self.point2;
        let area = u.cross(v).norm() / 2.0;
        if area < 1e-5 {
            return false;
        }
        return true;
    }
}

impl PartialEq for TriangleSTL {
    fn eq(&self, other: &Self) -> bool {
        return self.point0 == other.point0
            && self.point1 == other.point1
            && self.point2 == other.point2
            && self.normal == other.normal;
    }
}

impl Eq for TriangleSTL {}

impl fmt::Display for TriangleSTL {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return write!(
            f,
            "(p0: {}, p1: {}, p2: {}, normal: {})",
            self.point0, self.point1, self.point2, self.normal
        );
    }
}

fn get_factor_offset(
    min_vals: utils::Vec3f,
    max_vals: utils::Vec3f,
    size: f32,
    direction: &str,
) -> (f32, utils::Vec3f) {
    let mul_factor: f32;

    // normalize between 0 and total_dist in given direction
    if direction == "x" {
        mul_factor = size / (max_vals.x - min_vals.x);
    } else if direction == "y" {
        mul_factor = size / (max_vals.y - min_vals.y);
    } else if direction == "z" {
        mul_factor = size / (max_vals.z - min_vals.z);
    } else {
        panic!("Invalid direction {}", direction)
    }

    let offset = utils::Vec3f {
        x: min_vals.x,
        y: min_vals.y,
        z: min_vals.z,
    };
    return (mul_factor, offset);
}

fn get_triangles_min_max(triangles: &Vec<TriangleSTL>) -> (utils::Vec3f, utils::Vec3f) {
    let mut min_vals = utils::Vec3f {
        x: f32::MAX,
        y: f32::MAX,
        z: f32::MAX,
    };
    let mut max_vals = utils::Vec3f {
        x: f32::MIN,
        y: f32::MIN,
        z: f32::MIN,
    };
    for t in triangles {
        for p in [&t.point0, &t.point1, &t.point2] {
            if p.x < min_vals.x {
                min_vals.x = p.x;
            } else if p.x > max_vals.x {
                max_vals.x = p.x;
            }
            if p.y < min_vals.y {
                min_vals.y = p.y;
            } else if p.y > max_vals.y {
                max_vals.y = p.y;
            }
            if p.z < min_vals.z {
                min_vals.z = p.z;
            } else if p.z > max_vals.z {
                max_vals.z = p.z;
            }
        }
    }
    return (min_vals, max_vals);
}

pub fn normalize_triangles(
    triangles: &Vec<TriangleSTL>,
    size: f32,
    direction: &str,
) -> Vec<TriangleSTL> {
    let (min_vals, max_vals) = get_triangles_min_max(&triangles);
    let (mul_factor, offset) = get_factor_offset(min_vals, max_vals, size, direction);
    let mut normalized_triangles: Vec<TriangleSTL> = Vec::new();
    for t in triangles.iter() {
        let mut normalized_t = Clone::clone(t);
        normalized_t.normalize(mul_factor, offset);
        normalized_triangles.push(normalized_t);
    }
    return normalized_triangles;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stl::reader::read_stl;
    use crate::utils::almost_equal;
    use std::{path, string::String};

    fn check_normalization(
        orig_triangles: Vec<TriangleSTL>,
        norm_triangles: Vec<TriangleSTL>,
        size: f32,
    ) {
        for t in norm_triangles.iter() {
            for p in [&t.point0, &t.point1, &t.point2] {
                if p.x < 0f32 || p.x > size {
                    panic!("Point {} x is not between 0 and {}", p, size);
                }
            }
        }
        assert_eq!(norm_triangles.len(), orig_triangles.len());

        let max_x = norm_triangles
            .iter()
            .flat_map(|t: &TriangleSTL| Vec::from([t.point0.x, t.point1.x, t.point2.x]))
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();
        let min_x = norm_triangles
            .iter()
            .flat_map(|t: &TriangleSTL| Vec::from([t.point0.x, t.point1.x, t.point2.x]))
            .min_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();

        if !almost_equal(max_x, size) {
            panic!("Max is not same as {}", size);
        }
        if !almost_equal(min_x, 0f32) {
            panic!("Min is not same as 0");
        }
    }

    #[test]
    fn normalizes_stl_cube() {
        let str_filename = String::from("examples/stl/cube.stl");
        let filename = path::Path::new(str_filename.as_str()).to_owned();
        let size: f32 = 3.5;
        let triangles = read_stl(&filename);
        let norm_triangles = normalize_triangles(&triangles, size, "y");
        check_normalization(triangles, norm_triangles, size);
    }

    #[test]
    fn normalizes_stl_terrain() {
        let str_filename = String::from("examples/stl/terrain.stl");
        let filename = path::Path::new(str_filename.as_str()).to_owned();
        let size: f32 = 15.0;
        let triangles = read_stl(&filename);
        let norm_triangles = normalize_triangles(&triangles, size, "x");
        check_normalization(triangles, norm_triangles, size);
    }
}
