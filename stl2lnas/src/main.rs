pub mod lagrangian {
    pub mod format;
    pub mod join;
    pub mod save;
    pub mod triangle;
    pub mod vertice;
}

pub mod stl {
    pub mod reader;
    pub mod surfaces;
    pub mod triangle;
}
pub mod cfg;
pub mod utils;

use cfg::Args;
use clap::Parser;
use std::path;
use stl::surfaces::get_surfaces;

fn generate_lnas(args: &Args) {
    let (triangles, surfaces) = get_surfaces(&args.all_stls());

    let lagrangian_vertices = lagrangian::vertice::generate_lagrangian_vertices(&triangles);
    let lagrangian_triangles =
        lagrangian::triangle::generate_lagrangian_triangles(&lagrangian_vertices, &triangles);

    let (joined_vertices, joined_triangles) =
        lagrangian::join::join_information(&lagrangian_vertices, &lagrangian_triangles);
    let lnas_obj =
        lagrangian::format::get_lnas_obj_save(&joined_vertices, &joined_triangles, &surfaces);

    let lnas_filename = path::Path::new(&args.output);

    lagrangian::save::save_lnas(lnas_filename, &lnas_obj)
        .unwrap_or_else(|e| panic!("Saving lnas error. Error: {}", e));
}

fn main() {
    let args = Args::parse();

    if args.file.len() == 0 && args.dir.len() == 0 {
        println!("No file or dir to convert");
        return;
    }

    let lnas_filename = path::Path::new(&args.output);
    if lnas_filename.exists() {
        if !args.overwrite {
            panic!(
                "File '{}' already exists. Add '--overwrite' if you wish to overwrite it",
                args.output
            );
        } else {
            println!("Overwriting file...");
        }
    }

    if args.copy_stl {
        args.save_stl_to_output_folder()
            .unwrap_or_else(|e| println!("Unable to save STL in its output folder. Error: {}", e));
    }

    generate_lnas(&args);
    println!("Generated!");
}
