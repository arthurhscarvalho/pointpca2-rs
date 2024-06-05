use libm::acos;
use na::{DMatrix, MatrixView};
use num_traits::Pow;
use std::{f64::consts::PI, f64::EPSILON};

fn relative_difference(x: f64, y: f64) -> f64 {
    return 1. - (x - y).abs() / (x.abs() + y.abs() + EPSILON);
}

fn iter_relative_difference<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
) -> DMatrix<f64> {
    assert_eq!(x.nrows(), y.nrows(), "Matrices must have the same shape.");
    assert_eq!(x.ncols(), y.ncols(), "Matrices must have the same shape.");
    let nrows = x.nrows();
    let ncols = x.ncols();
    let mut result = DMatrix::zeros(nrows, ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            result[(i, j)] = relative_difference(x[(i, j)], y[(i, j)]);
        }
    }
    result
}

fn covariance_differences<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
    z: &'a MatrixView<f64, T, T>,
) -> DMatrix<f64> {
    assert_eq!(x.nrows(), y.nrows(), "Matrices must have the same shape.");
    assert_eq!(x.ncols(), y.ncols(), "Matrices must have the same shape.");
    assert_eq!(x.nrows(), z.nrows(), "Matrices must have the same shape.");
    assert_eq!(x.ncols(), z.ncols(), "Matrices must have the same shape.");
    let nrows = x.nrows();
    let ncols = x.ncols();
    let mut result = DMatrix::zeros(nrows, ncols);
    let mut x_ij: f64;
    let mut y_ij: f64;
    let mut z_ij: f64;
    for i in 0..nrows {
        for j in 0..ncols {
            x_ij = x[(i, j)];
            y_ij = y[(i, j)];
            z_ij = z[(i, j)];
            result[(i, j)] =
                (x_ij.sqrt() * y_ij.sqrt() - z_ij).abs() / (x_ij.sqrt() * y_ij.sqrt() + EPSILON);
        }
    }
    result
}

fn textural_variance_sum<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
) -> DMatrix<f64> {
    assert_eq!(x.nrows(), y.nrows(), "Matrices must have the same shape.");
    assert_eq!(x.ncols(), y.ncols(), "Matrices must have the same shape.");
    let nrows = x.nrows();
    let ncols = x.ncols();
    let mut result = DMatrix::zeros(nrows, 1);
    for i in 0..nrows {
        let mut x_sum: f64 = 0.;
        let mut y_sum: f64 = 0.;
        for j in 0..ncols {
            x_sum += x[(i, j)];
            y_sum += y[(i, j)];
        }
        result[(i, 0)] = relative_difference(x_sum, y_sum);
    }
    result
}

fn omnivariance_differences<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
) -> DMatrix<f64> {
    assert_eq!(x.nrows(), y.nrows(), "Matrices must have the same shape.");
    assert_eq!(x.ncols(), y.ncols(), "Matrices must have the same shape.");
    let nrows = x.nrows();
    let ncols = x.ncols();
    let mut result = DMatrix::zeros(nrows, 1);
    for i in 0..nrows {
        let mut x_prod: f64 = 1.;
        let mut y_prod: f64 = 1.;
        for j in 0..ncols {
            x_prod *= x[(i, j)];
            y_prod *= y[(i, j)];
        }
        result[(i, 0)] = relative_difference(x_prod.cbrt(), y_prod.cbrt());
    }
    result
}

fn entropy<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
) -> DMatrix<f64> {
    assert_eq!(x.nrows(), y.nrows(), "Matrices must have the same shape.");
    assert_eq!(x.ncols(), y.ncols(), "Matrices must have the same shape.");
    let nrows = x.nrows();
    let ncols = x.ncols();
    let mut result = DMatrix::zeros(nrows, 1);
    for i in 0..nrows {
        let mut x_entropy: f64 = 0.;
        let mut y_entropy: f64 = 0.;
        for j in 0..ncols {
            x_entropy += x[(i, j)] * (x[(i, j)] + EPSILON).ln();
            y_entropy += y[(i, j)] * (y[(i, j)] + EPSILON).ln();
        }
        result[(i, 0)] = relative_difference(x_entropy, y_entropy);
    }
    result
}

fn euclidean_distances<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
) -> DMatrix<f64> {
    assert_eq!(x.nrows(), y.nrows(), "Matrices must have the same shape.");
    assert_eq!(x.ncols(), y.ncols(), "Matrices must have the same shape.");
    let nrows = x.nrows();
    let ncols = x.ncols();
    let mut result = DMatrix::zeros(nrows, 1);
    for i in 0..nrows {
        let mut dists: f64 = 0.;
        for j in 0..ncols {
            dists += (x[(i, j)] - y[(i, j)]).pow(2);
        }
        result[(i, 0)] = dists;
    }
    result
}

fn vector_projected_distances<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
    col: usize,
) -> DMatrix<f64> {
    assert_eq!(x.nrows(), y.nrows(), "Matrices must have the same shape.");
    assert_eq!(x.ncols(), y.ncols(), "Matrices must have the same shape.");
    let ncols = 1;
    let nrows = x.nrows();
    let mut result = DMatrix::zeros(nrows, ncols);
    for i in 0..nrows {
        result[(i, 0)] = (x[(i, col)] - y[(i, col)]).abs();
    }
    result
}

fn point_projected_distances<'a, T: na::Dim>(x: &'a MatrixView<f64, T, T>) -> DMatrix<f64> {
    let ncols = 2;
    let nrows = x.nrows();
    let mut result = DMatrix::zeros(nrows, ncols);
    for i in 0..nrows {
        result[(i, 0)] = x[(i, 0)].abs();
        result[(i, 1)] = x[(i, 1)].abs();
    }
    result
}

fn point_to_centroid_distances<'a, T: na::Dim>(x: &'a MatrixView<f64, T, T>) -> DMatrix<f64> {
    let ncols = 1;
    let nrows = x.nrows();
    let mut result = DMatrix::zeros(nrows, ncols);
    for i in 0..nrows {
        let mut dists: f64 = 0.;
        for j in 0..ncols {
            dists += x[(i, j)].pow(2);
        }
        result[(i, 0)] = dists.sqrt();
    }
    result
}

fn anisotropy_planarity_linearity<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
    col1: usize,
    col2: usize,
) -> DMatrix<f64> {
    assert_eq!(x.nrows(), y.nrows(), "Matrices must have the same shape.");
    assert_eq!(x.ncols(), y.ncols(), "Matrices must have the same shape.");
    let nrows = x.nrows();
    let mut result = DMatrix::zeros(nrows, 1);
    for i in 0..nrows {
        let x_diff = (x[(i, col1)] - x[(i, col2)]) / x[(i, 0)];
        let y_diff = (y[(i, col1)] - y[(i, col2)]) / y[(i, 0)];
        result[(i, 0)] = relative_difference(x_diff, y_diff);
    }
    result
}

fn surface_variation<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
) -> DMatrix<f64> {
    assert_eq!(x.nrows(), y.nrows(), "Matrices must have the same shape.");
    assert_eq!(x.ncols(), y.ncols(), "Matrices must have the same shape.");
    let nrows = x.nrows();
    let ncols = x.ncols();
    let mut result = DMatrix::zeros(nrows, 1);
    for i in 0..nrows {
        let mut x_sum: f64 = 0.;
        let mut y_sum: f64 = 0.;
        for j in 0..ncols {
            x_sum += x[(i, j)];
            y_sum += y[(i, j)];
        }
        result[(i, 0)] = relative_difference(x[(i, 2)] / x_sum, y[(i, 2)] / y_sum);
    }
    result
}

fn sphericity<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
) -> DMatrix<f64> {
    assert_eq!(x.nrows(), y.nrows(), "Matrices must have the same shape.");
    assert_eq!(x.ncols(), y.ncols(), "Matrices must have the same shape.");
    let nrows = x.nrows();
    let mut result = DMatrix::zeros(nrows, 1);
    for i in 0..nrows {
        result[(i, 0)] = relative_difference(x[(i, 2)] / x[(i, 0)], y[(i, 2)] / y[(i, 0)]);
    }
    result
}

fn angular_similarity<'a, T: na::Dim>(x: &'a MatrixView<f64, T, T>) -> DMatrix<f64> {
    let nrows = x.nrows();
    let mut result = DMatrix::zeros(nrows, 1);
    for i in 0..nrows {
        let numerator = x[(i, 1)];
        let mut denominator: f64 = x[(i, 1)].pow(2);
        denominator = denominator.sqrt();
        result[(i, 0)] = 1. - 2. * acos(numerator / denominator).abs() / PI;
    }
    result
}

fn parallelity<'a, T: na::Dim>(x: &'a MatrixView<f64, T, T>, col: usize) -> DMatrix<f64> {
    let nrows = x.nrows();
    let mut result = DMatrix::zeros(nrows, 1);
    for (i, num) in x.column(col).iter().enumerate() {
        result[i] = *num;
    }
    result
}

pub fn compute_predictors<'a>(local_features: &'a DMatrix<f64>) -> DMatrix<f64> {
    let nrows = local_features.nrows();
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
    predictors
        .view_mut((0, 0), (nrows, 3))
        .copy_from(&iter_relative_difference(&colors_mean_a, &colors_mean_b));
    predictors
        .view_mut((0, 3), (nrows, 3))
        .copy_from(&iter_relative_difference(
            &colors_variance_a,
            &colors_variance_b,
        ));
    predictors
        .view_mut((0, 6), (nrows, 3))
        .copy_from(&covariance_differences(
            &colors_variance_a,
            &colors_variance_b,
            &colors_covariance_ab,
        ));
    predictors
        .view_mut((0, 9), (nrows, 1))
        .copy_from(&textural_variance_sum(
            &colors_variance_a,
            &colors_variance_b,
        ));
    predictors
        .view_mut((0, 10), (nrows, 1))
        .copy_from(&omnivariance_differences(
            &colors_variance_a,
            &colors_variance_b,
        ));
    predictors
        .view_mut((0, 11), (nrows, 1))
        .copy_from(&entropy(&colors_variance_a, &colors_variance_b));
    // Geometric predictors
    predictors
        .view_mut((0, 12), (nrows, 1))
        .copy_from(&euclidean_distances(&projection_a_to_a, &projection_b_to_a));
    predictors
        .view_mut((0, 13), (nrows, 1))
        .copy_from(&vector_projected_distances(
            &projection_a_to_a,
            &projection_b_to_a,
            0,
        ));
    predictors
        .view_mut((0, 14), (nrows, 1))
        .copy_from(&vector_projected_distances(
            &projection_a_to_a,
            &projection_b_to_a,
            1,
        ));
    predictors
        .view_mut((0, 15), (nrows, 1))
        .copy_from(&vector_projected_distances(
            &projection_a_to_a,
            &projection_b_to_a,
            2,
        ));
    predictors
        .view_mut((0, 16), (nrows, 2))
        .copy_from(&point_projected_distances(&projection_a_to_a));
    predictors
        .view_mut((0, 18), (nrows, 1))
        .copy_from(&point_to_centroid_distances(&projection_b_to_a));
    predictors
        .view_mut((0, 19), (nrows, 2))
        .copy_from(&point_projected_distances(&projection_b_to_a));
    predictors
        .view_mut((0, 21), (nrows, 1))
        .copy_from(&point_to_centroid_distances(&points_mean_b));
    predictors
        .view_mut((0, 22), (nrows, 2))
        .copy_from(&point_projected_distances(&points_mean_b));
    predictors
        .view_mut((0, 24), (nrows, 3))
        .copy_from(&iter_relative_difference(
            &points_variance_a,
            &points_variance_b,
        ));
    predictors
        .view_mut((0, 27), (nrows, 3))
        .copy_from(&covariance_differences(
            &points_variance_a,
            &points_variance_b,
            &points_covariance_ab,
        ));
    predictors
        .view_mut((0, 30), (nrows, 1))
        .copy_from(&omnivariance_differences(
            &points_variance_a,
            &points_variance_b,
        ));
    predictors
        .view_mut((0, 31), (nrows, 1))
        .copy_from(&entropy(&points_variance_a, &points_variance_b));
    predictors
        .view_mut((0, 32), (nrows, 1))
        .copy_from(&anisotropy_planarity_linearity(
            &points_variance_a,
            &points_variance_b,
            0,
            2,
        ));
    predictors
        .view_mut((0, 33), (nrows, 1))
        .copy_from(&anisotropy_planarity_linearity(
            &points_variance_a,
            &points_variance_b,
            1,
            2,
        ));
    predictors
        .view_mut((0, 34), (nrows, 1))
        .copy_from(&anisotropy_planarity_linearity(
            &points_variance_a,
            &points_variance_b,
            0,
            1,
        ));
    predictors
        .view_mut((0, 35), (nrows, 1))
        .copy_from(&surface_variation(&points_variance_a, &points_variance_b));
    predictors
        .view_mut((0, 36), (nrows, 1))
        .copy_from(&sphericity(&points_variance_a, &points_variance_b));
    predictors
        .view_mut((0, 37), (nrows, 1))
        .copy_from(&angular_similarity(&points_eigenvectors_b_y));
    predictors
        .view_mut((0, 38), (nrows, 1))
        .copy_from(&parallelity(&points_eigenvectors_b_x, 0));
    predictors
        .view_mut((0, 39), (nrows, 1))
        .copy_from(&parallelity(&points_eigenvectors_b_z, 2));
    predictors
}
