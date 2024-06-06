extern crate kiddo;
extern crate libm;
extern crate nalgebra as na;
extern crate ordered_float;
extern crate ply_rs;

mod features;
mod knn_search;
mod ply_manager;
mod predictors;
mod preprocessing;
mod utils;

fn main() {
    let path = "/home/arthurc/romanoillamp_vox10.ply";
    let (points_a, colors_a) = ply_manager::read_ply_as_matrix(path);
    for i in 0..10 {
        for j in 0..3 {
            print!("{} ", points_a[(i, j)]);
        }
        for j in 0..3 {
            print!("{} ", colors_a[(i, j)]);
        }
        println!("");
    }
}
