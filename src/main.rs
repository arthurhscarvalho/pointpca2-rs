extern crate kiddo;
extern crate libm;
extern crate nalgebra as na;
extern crate ordered_float;
extern crate ply_rs;

mod features;
mod knn_search;
mod ply_manager;
mod pooling;
mod predictors;
mod preprocessing;
mod utils;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let search_size = 81;
    println!("Reading ply");
    let (points_a, colors_a) = ply_manager::read_ply_as_matrix("/home/arthurc/redandblack_vox10_1550.ply");
    let (points_b, colors_b) = ply_manager::read_ply_as_matrix("/home/arthurc/tmc13_redandblack_vox10_1550_dec_geom04_text04_octree-predlift.ply");
    println!("Preprocessing");
    let (points_a, colors_a) = preprocessing::preprocess_point_cloud(&points_a, &colors_a);
    let (points_b, colors_b) = preprocessing::preprocess_point_cloud(&points_b, &colors_b);
    println!("Performing knn search");
    let knn_indices_a = knn_search::knn_search(&points_a, &points_a, search_size);
    let knn_indices_b = knn_search::knn_search(&points_a, &points_b, search_size);
    println!("Computing local features");
    let local_features = features::compute_features(
        &points_a,
        &colors_a,
        &points_b,
        &colors_b,
        &knn_indices_a,
        &knn_indices_b,
        search_size
    );
    println!("Computing predictors");
    let predictors_result = predictors::compute_predictors(&local_features);
    println!("Pooling predictors");
    let pooled_predictors = pooling::mean_pooling(&predictors_result);
    println!("Predictors:");
    for col in pooled_predictors.iter() {
        print!("{:.2} ", *col);
    }
    println!("");
}
