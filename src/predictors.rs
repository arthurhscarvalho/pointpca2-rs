use std::f64::EPSILON;

use na::DMatrix;

fn relative_difference(x: f64, y: f64) -> f64 {
    return 1. - (x - y).abs() / (x.abs() + y.abs() + EPSILON);
}

pub fn compute_predictors<'a>(local_features: &'a DMatrix<f64>) -> DMatrix<f64> {
    let nrows = local_features.nrows();
    let ncols = local_features.ncols();
    let projection_a_to_a = local_features.view((0, 0), (nrows, 3));
    let projection_b_to_a = local_features.view((0, 3), (nrows, 3));
    let colors_mean_a = local_features.view((0, 6), (nrows, 3));
    let points_mean_b = local_features.view((0, 9), (nrows, 3));
    let colors_mean_b = local_features.view((0, 12), (nrows, 3));
    let points_variance_a = local_features.view((0, 15), (nrows, 3));
    let colors_variance_a = local_features.view((0, 18), (nrows, 3));
    let points_variance_b = local_features.view((0, 21), (nrows, 3));
    let colors_variance_b = local_features.view((0, 24), (nrows, 3));
    let points_covariance_ab = local_features.view((0, 27), (nrows, 3));
    let colors_covariance_ab = local_features.view((0, 30), (nrows, 3));
    let points_eigenvectors_b_x = local_features.view((0, 33), (nrows, 3));
    let points_eigenvectors_b_y = local_features.view((0, 36), (nrows, 3));
    let points_eigenvectors_b_z = local_features.view((0, 39), (nrows, 3));
    let mut predictors = DMatrix::zeros(nrows, 40);
    predictors.fill(f64::NAN);

    // Textural predictors
    // To-Do
    // . . .

    // Geometric predictors
    // To-Do
    // . . .

    predictors
}