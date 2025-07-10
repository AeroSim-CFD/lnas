use crate::utils::create_folder;
use clap::Parser;
use std::error::Error;
use std::{collections::HashMap, fs, path, string::String};

#[derive(Parser)]
#[command(author, version, about, long_about=None)] // Read from `Cargo.toml`
pub struct Args {
    /// Directories to use
    #[arg(short, long, num_args(0..), help = "Directories with STLs to use")]
    pub dir: Vec<String>,

    /// Files to use
    #[arg(short, long, num_args(0..), help = "STL filenames to use")]
    pub file: Vec<String>,

    /// Output folder
    #[arg(short, long, help = "Output filename for .lnas")]
    pub output: String,

    /// Overwrite existing files
    #[arg(long, action, help = "Overwrite existing files")]
    pub overwrite: bool,

    /// Copy STLs or not
    #[arg(long, action, help = "Copy STLs to output folder")]
    pub copy_stl: bool,
}

impl Args {
    pub fn folder_stls(&self, dir: &str) -> Vec<path::PathBuf> {
        let paths = fs::read_dir(dir).unwrap();
        let mut all_paths: Vec<path::PathBuf> = Vec::new();
        for path in paths {
            let full_path = path.unwrap().path();
            if full_path.to_str().unwrap().ends_with(".stl") && full_path.is_file() {
                all_paths.push(full_path.to_path_buf());
            }
        }
        return all_paths;
    }

    pub fn all_stls(&self) -> HashMap<String, path::PathBuf> {
        let mut stls: HashMap<String, path::PathBuf> = HashMap::new();
        for foldername in self.dir.iter() {
            let paths = self.folder_stls(foldername);
            for p in paths.iter() {
                let name = p
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .strip_suffix(".stl")
                    .unwrap();
                if stls.contains_key(name) {
                    panic!("Repeated name {} in folder STLs", name);
                }
                stls.insert(name.to_string(), p.to_owned());
            }
        }
        for str_filename in self.file.iter() {
            let filename = path::Path::new(str_filename.as_str());
            let name = filename
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .strip_suffix(".stl")
                .unwrap();
            if stls.contains_key(name) {
                panic!("Repeated name {} in file STLs", name);
            }
            stls.insert(name.to_owned(), filename.to_owned());
        }
        return stls;
    }

    pub fn save_stl_to_output_folder(&self) -> Result<(), Box<dyn Error>> {
        let mut str_foldername_stl = self.output.to_owned();
        str_foldername_stl.push_str(".stls");
        let foldername = path::Path::new(str_foldername_stl.as_str());
        create_folder(foldername)?;
        for (stl_name, stl_filename) in self.all_stls().iter() {
            fs::copy(stl_filename, foldername.join(format!("{}.stl", stl_name)))?;
        }
        return Ok(());
    }
}
