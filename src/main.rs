extern crate nalgebra as na;
extern crate ordered_float;
extern crate kiddo;
extern crate libm;
extern crate ply_rs;

use ply_rs as ply;
use knn_search::knn_search;
use ordered_float::OrderedFloat;
use std::collections::HashSet;

mod features;
mod predictors;
mod preprocessing;
mod utils;
mod knn_search;

fn main() {
    let path = "/home/arthurc/romanoillamp_vox10.ply";
    let pc1 = utils::read_ply(path);
    println!("Ply header: {:#?}", pc1.header);
}
