extern crate kiddo;
extern crate libm;
extern crate nalgebra as na;
extern crate ordered_float;
extern crate ply_rs;

use na::DMatrix;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod features;
mod knn_search;
mod ply_manager;
mod pooling;
mod predictors;
mod preprocessing;
mod utils;

#[pyfunction]
fn pointpca2(
    _py: Python,
    points_a: Vec<Vec<f64>>,
    colors_a: Vec<Vec<u8>>,
    points_b: Vec<Vec<f64>>,
    colors_b: Vec<Vec<u8>>,
    search_size: usize,
    verbose: bool,
) -> &PyArray1<f64> {
    let points_a = DMatrix::from_vec(
        points_a.len(),
        points_a[0].len(),
        points_a.into_iter().flat_map(|v| v.into_iter()).collect(),
    );
    let colors_a = DMatrix::from_vec(
        colors_a.len(),
        colors_a[0].len(),
        colors_a.into_iter().flat_map(|v| v.into_iter()).collect(),
    );
    let points_b = DMatrix::from_vec(
        points_b.len(),
        points_b[0].len(),
        points_b.into_iter().flat_map(|v| v.into_iter()).collect(),
    );
    let colors_b = DMatrix::from_vec(
        colors_b.len(),
        colors_b[0].len(),
        colors_b.into_iter().flat_map(|v| v.into_iter()).collect(),
    );
    if verbose {
        println!("Preprocessing");
    }
    let (points_a, colors_a) = preprocessing::preprocess_point_cloud(&points_a, &colors_a);
    let (points_b, colors_b) = preprocessing::preprocess_point_cloud(&points_b, &colors_b);
    if verbose {
        println!("Performing knn search");
    }
    let knn_indices_a = knn_search::knn_search(&points_a, &points_a, search_size);
    let knn_indices_b = knn_search::knn_search(&points_b, &points_a, search_size);
    if verbose {
        println!("Computing local features");
    }
    let local_features = features::compute_features(
        &points_a,
        &colors_a,
        &points_b,
        &colors_b,
        &knn_indices_a,
        &knn_indices_b,
        search_size,
    );
    if verbose {
        println!("Computing predictors");
    }
    let predictors_result = predictors::compute_predictors(&local_features);
    if verbose {
        println!("Pooling predictors");
    }
    let pooled_predictors = pooling::mean_pooling(&predictors_result);
    if verbose {
        println!("Predictors:");
        for col in pooled_predictors.iter() {
            print!("{:.4} ", *col);
        }
        println!("");
    }
    let py_array = PyArray1::from_iter(_py, pooled_predictors.iter().cloned());
    py_array
}

#[pymodule]
fn pointpca2_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pointpca2, m)?)?;
    Ok(())
}
