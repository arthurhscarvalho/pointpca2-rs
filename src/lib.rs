extern crate kd_tree;
extern crate libm;
extern crate nalgebra as na;
extern crate ordered_float;
extern crate ply_rs;
extern crate rayon;

mod eigenvectors;
mod features;
mod knn_search;
pub mod ply_manager;
mod predictors;
mod preprocessing;
mod spatial_metrics;
mod utils;

pub fn compute_pointpca2<'a>(
    points_a: na::DMatrix<f64>,
    colors_a: na::DMatrix<u8>,
    points_b: na::DMatrix<f64>,
    colors_b: na::DMatrix<u8>,
    search_size: usize,
    verbose: bool,
) -> na::Matrix1xX<f64> {
    utils::print_if_verbose("Preprocessing", &verbose);
    let (points_a, colors_a) = preprocessing::preprocess_point_cloud(&points_a, &colors_a);
    let (points_b, colors_b) = preprocessing::preprocess_point_cloud(&points_b, &colors_b);
    utils::print_if_verbose("Computing local features", &verbose);
    let local_features = features::compute_features(
        points_a,
        colors_a,
        points_b,
        colors_b,
        search_size,
    );
    utils::print_if_verbose("Computing predictors", &verbose);
    let predictors_result = predictors::compute_predictors(local_features);
    predictors_result
}
