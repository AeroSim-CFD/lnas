#[macro_export]
macro_rules! assert_almost_equal {
    ($x:expr, $y:expr, $d:expr) => {
        if (($x - $y).abs() > $d) {
            let error_str = format!("{:.4e} != {:.4e} by more than {:.4e}", $x, $y, $d);
            panic!("{}", error_str);
        };
    };
}

use serde::Serialize;
use std::{cmp::Ordering, convert::TryInto, error::Error, fmt, fs, hash, ops, path};

const PREC_DIGITS: i32 = 5i32;

fn truncate_float_to_int(f: f32, n_digits: i32) -> i32 {
    let y = (f * 10f32.powi(n_digits)).round() as i32;
    return y;
}

// I don't know how to check endianess, but this is the way that it works
pub fn bytes_to_u32_be(b: &[u8; 4]) -> u32 {
    return u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
}

pub fn bytes_to_f32_be(b: &[u8; 4]) -> f32 {
    return f32::from_be_bytes([b[0], b[1], b[2], b[3]]);
}

pub fn bytes_to_u32_le(b: &[u8; 4]) -> u32 {
    return u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
}

pub fn bytes_to_f32_le(b: &[u8; 4]) -> f32 {
    return f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
}

pub fn create_folder(foldername: &path::Path) -> Result<(), Box<dyn Error>> {
    if foldername.exists() {
        return Ok(());
    }
    fs::create_dir_all(foldername)?;
    return Ok(());
}

pub fn create_folder_for_filename(filename: &path::Path) -> Result<(), Box<dyn Error>> {
    if filename.parent().is_some() {
        if filename.parent().unwrap().exists() {
            return Ok(());
        }
        fs::create_dir_all(filename.parent().unwrap())?;
    }
    return Ok(());
}

#[derive(Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Vec3u {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Vec3u {
    pub fn to_le_bytes_as_u32(self) -> [u8; 12] {
        return [
            (self.x as u32).to_le_bytes(),
            (self.y as u32).to_le_bytes(),
            (self.z as u32).to_le_bytes(),
        ]
        .concat()
        .try_into()
        .unwrap_or_else(|v: Vec<u8>| panic!("Expected a Vec of length 12, got {}", v.len()));
    }

    pub fn from_bytes_be(vec_bytes: &Vec<u8>) -> Vec3u {
        return Vec3u {
            x: bytes_to_u32_be(vec_bytes[..4].try_into().expect("Invalid point bytes")) as u32,
            y: bytes_to_u32_be(vec_bytes[4..8].try_into().expect("Invalid point bytes")) as u32,
            z: bytes_to_u32_be(vec_bytes[8..12].try_into().expect("Invalid point bytes")) as u32,
        };
    }

    pub fn from_bytes_le(vec_bytes: &Vec<u8>) -> Vec3u {
        return Vec3u {
            x: bytes_to_u32_le(vec_bytes[..4].try_into().expect("Invalid point bytes")) as u32,
            y: bytes_to_u32_le(vec_bytes[4..8].try_into().expect("Invalid point bytes")) as u32,
            z: bytes_to_u32_le(vec_bytes[8..12].try_into().expect("Invalid point bytes")) as u32,
        };
    }
}

impl fmt::Display for Vec3u {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return write!(f, "({}, {}, {})", self.x, self.y, self.z);
    }
}

/// Algebraic Point
#[derive(Clone, Copy, Serialize, Debug)]
pub struct Vec3f {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3f {
    pub fn norm(self) -> f32 {
        return (self.dot(self)).sqrt();
    }

    pub fn normalize(&mut self) {
        self.divide(self.norm());
    }

    pub fn dot(self, other: Vec3f) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn transform(&mut self, factor: f32, offset: Self) {
        *self = *self - offset;
        self.multiply(factor)
    }

    pub fn divide(&mut self, denominator: f32) {
        self.x /= denominator;
        self.y /= denominator;
        self.z /= denominator;
    }

    pub fn multiply(&mut self, factor: f32) {
        self.x *= factor;
        self.y *= factor;
        self.z *= factor;
    }

    pub fn cross(self, other: Self) -> Self {
        return Vec3f {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        };
    }

    pub fn abs(self) -> Self {
        let mut x = self.x;
        if x < 0f32 {
            x = -x;
        }
        let mut y = self.y;
        if y < 0f32 {
            y = -y;
        }
        let mut z = self.z;
        if z < 0f32 {
            z = -z;
        }
        return Vec3f { x, y, z };
    }

    pub fn to_le_bytes_as_f32(self) -> Vec<u8> {
        return [
            (self.x as f32).to_le_bytes(),
            (self.y as f32).to_le_bytes(),
            (self.z as f32).to_le_bytes(),
        ]
        .concat();
        // .unwrap_or_else(|v: Vec<u8>| panic!("Expected a Vec of length 12, got {}", v.len()));
    }

    pub fn from_bytes_be(vec_bytes: &Vec<u8>) -> Vec3f {
        return Vec3f {
            x: bytes_to_f32_be(vec_bytes[..4].try_into().expect("Invalid point bytes")) as f32,
            y: bytes_to_f32_be(vec_bytes[4..8].try_into().expect("Invalid point bytes")) as f32,
            z: bytes_to_f32_be(vec_bytes[8..12].try_into().expect("Invalid point bytes")) as f32,
        };
    }

    pub fn from_bytes_le(vec_bytes: &Vec<u8>) -> Vec3f {
        return Vec3f {
            x: bytes_to_f32_le(vec_bytes[..4].try_into().expect("Invalid point bytes")) as f32,
            y: bytes_to_f32_le(vec_bytes[4..8].try_into().expect("Invalid point bytes")) as f32,
            z: bytes_to_f32_le(vec_bytes[8..12].try_into().expect("Invalid point bytes")) as f32,
        };
    }
}

pub fn almost_equal(x: f32, y: f32) -> bool {
    return truncate_float_to_int(x, PREC_DIGITS) == truncate_float_to_int(y, PREC_DIGITS);
}

impl fmt::Display for Vec3f {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return write!(f, "({:.2}, {:.2}, {:.2})", self.x, self.y, self.z);
    }
}

impl PartialEq for Vec3f {
    fn eq(&self, other: &Self) -> bool {
        return almost_equal(self.x, other.x)
            && almost_equal(self.y, other.y)
            && almost_equal(self.z, other.z);
    }
}

impl PartialOrd for Vec3f {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        return Some(self.cmp(other));
    }
}

impl Eq for Vec3f {}

impl Ord for Vec3f {
    fn cmp(&self, other: &Self) -> Ordering {
        // check if all numbers are almost equal, if so, points are equal
        if almost_equal(self.x, other.x)
            && almost_equal(self.y, other.y)
            && almost_equal(self.z, other.z)
        {
            return Ordering::Equal;
        }
        // check for dimensions difference, priority is (x, y, z)
        let vals_cmps = [(self.x, other.x), (self.y, other.y), (self.z, other.z)];
        for v in vals_cmps.iter() {
            if !almost_equal(v.0, v.1) {
                if v.0 > v.1 {
                    return Ordering::Greater;
                } else {
                    return Ordering::Less;
                }
            }
        }
        panic!(
            "Invalid comparison between {} and {}. Implementation error",
            self, other
        );
    }
}

impl ops::Add for Vec3f {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        return Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        };
    }
}

impl ops::AddAssign for Vec3f {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl ops::Sub for Vec3f {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl ops::Div for Vec3f {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        return Self {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
        };
    }
}

impl ops::Mul for Vec3f {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        return Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        };
    }
}

impl hash::Hash for Vec3f {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        // Truncates value up to 5 digits after comma, then hashes it
        let n_digits = PREC_DIGITS;
        let (xt, yt, zt) = (
            truncate_float_to_int(self.x, n_digits),
            truncate_float_to_int(self.y, n_digits),
            truncate_float_to_int(self.z, n_digits),
        );
        xt.hash(state);
        yt.hash(state);
        zt.hash(state);
    }
}
