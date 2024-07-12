use libm::acos;
use na::{DMatrix, MatrixView};
use num_traits::Pow;
use std::{f64::consts::PI, f64::EPSILON};

fn relative_difference(x: f64, y: f64) -> f64 {
    1. - (x - y).abs() / (x.abs() + y.abs() + EPSILON)
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

pub fn omnivariance_differences<'a, T: na::Dim>(
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
        result[(i, 0)] = dists.sqrt();
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
        result[(i, 0)] = x[(i, 1)].abs();
        result[(i, 1)] = x[(i, 2)].abs();
    }
    result
}

pub fn point_to_centroid_distances<'a, T: na::Dim>(x: &'a MatrixView<f64, T, T>) -> DMatrix<f64> {
    let ncols = 1;
    let nrows = x.nrows();
    let mut result = DMatrix::zeros(nrows, ncols);
    for i in 0..nrows {
        let mut dists: f64 = 0.;
        for j in 0..x.ncols() {
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
        let x_diff = (x[(i, col1)] - x[(i, col2)]) / (x[(i, 0)] + EPSILON);
        let y_diff = (y[(i, col1)] - y[(i, col2)]) / (y[(i, 0)] + EPSILON);
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
        result[(i, 0)] =
            relative_difference(x[(i, 2)] / (x_sum + EPSILON), y[(i, 2)] / (y_sum + EPSILON));
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
        result[(i, 0)] = relative_difference(
            x[(i, 2)] / (x[(i, 0)] + EPSILON),
            y[(i, 2)] / (y[(i, 0)] + EPSILON),
        );
    }
    result
}

pub fn angular_similarity<'a, T: na::Dim>(x: &'a MatrixView<f64, T, T>) -> DMatrix<f64> {
    let nrows = x.nrows();
    let mut result = DMatrix::zeros(nrows, 1);
    for i in 0..nrows {
        let numerator = x[(i, 1)];
        let (mut a, mut b, mut c) = (x[(i, 0)], x[(i, 1)], x[(i, 2)]);
        (a, b, c) = (a.pow(2), b.pow(2), c.pow(2));
        let denominator: f64 = (a + b + c).sqrt() + EPSILON;
        result[(i, 0)] = 1. - 2. * acos((numerator / denominator).abs()) / PI;
    }
    result
}

fn parallelity<'a, T: na::Dim>(x: &'a MatrixView<f64, T, T>, col: usize) -> DMatrix<f64> {
    let nrows = x.nrows();
    let mut result = DMatrix::zeros(nrows, 1);
    for (i, num) in x.column(col).iter().enumerate() {
        result[i] = 1. - *num;
    }
    result
}

pub fn compute_predictors(local_features: DMatrix<f64>) -> DMatrix<f64> {
    let projection_a_to_a = local_features.columns(0, 3);
    let projection_b_to_a = local_features.columns(3, 3);
    let colors_mean_a = local_features.columns(6, 3);
    let points_mean_b = local_features.columns(9, 3);
    let colors_mean_b = local_features.columns(12, 3);
    let points_variance_a = local_features.columns(15, 3);
    let colors_variance_a = local_features.columns(18, 3);
    let points_variance_b = local_features.columns(21, 3);
    let colors_variance_b = local_features.columns(24, 3);
    let points_covariance_ab = local_features.columns(27, 3);
    let colors_covariance_ab = local_features.columns(30, 3);
    let points_eigenvectors_b_x = local_features.columns(33, 3);
    let points_eigenvectors_b_y = local_features.columns(36, 3);
    let points_eigenvectors_b_z = local_features.columns(39, 3);
    let nrows = local_features.nrows();
    let ncols = 40;
    let mut predictors = DMatrix::zeros(nrows, ncols);
    /*
        Textural predictors
    */
    // Relative difference in mean color values
    predictors
        .columns_mut(0, 3)
        .copy_from(&iter_relative_difference(&colors_mean_a, &colors_mean_b));
    // Relative difference in color variance
    predictors
        .columns_mut(3, 3)
        .copy_from(&iter_relative_difference(
            &colors_variance_a,
            &colors_variance_b,
        ));
    // Covariance differences between color variances
    predictors
        .columns_mut(6, 3)
        .copy_from(&covariance_differences(
            &colors_variance_a,
            &colors_variance_b,
            &colors_covariance_ab,
        ));
    // Sum of variances of textures
    predictors.column_mut(9).copy_from(&textural_variance_sum(
        &colors_variance_a,
        &colors_variance_b,
    ));
    // Relative difference in omnivariance of textures
    predictors
        .column_mut(10)
        .copy_from(&omnivariance_differences(
            &colors_variance_a,
            &colors_variance_b,
        ));
    // Entropy of textures
    predictors
        .column_mut(11)
        .copy_from(&entropy(&colors_variance_a, &colors_variance_b));
    /*
        Geometric predictors
    */
    // Euclidean distance between distorted and reference points (error vector)
    predictors
        .column_mut(12)
        .copy_from(&euclidean_distances(&projection_a_to_a, &projection_b_to_a));
    // Projected distances of vectors between distorted and reference points from reference planes
    predictors
        .column_mut(13)
        .copy_from(&vector_projected_distances(
            &projection_a_to_a,
            &projection_b_to_a,
            0,
        ));
    predictors
        .column_mut(14)
        .copy_from(&vector_projected_distances(
            &projection_a_to_a,
            &projection_b_to_a,
            1,
        ));
    predictors
        .column_mut(15)
        .copy_from(&vector_projected_distances(
            &projection_a_to_a,
            &projection_b_to_a,
            2,
        ));
    // Projected distances of reference points from reference planes
    predictors
        .columns_mut(16, 2)
        .copy_from(&point_projected_distances(&projection_a_to_a));
    // Euclidean distance between distorted point and reference centroid
    predictors
        .column_mut(18)
        .copy_from(&point_to_centroid_distances(&projection_b_to_a));
    // Projected distances of distorted point from reference planes
    predictors
        .columns_mut(19, 2)
        .copy_from(&point_projected_distances(&projection_b_to_a));
    // Euclidean distance between distorted centroid and reference centroid
    predictors
        .column_mut(21)
        .copy_from(&point_to_centroid_distances(&points_mean_b));
    // Projected distances of distorted centroid from reference planes
    predictors
        .columns_mut(22, 2)
        .copy_from(&point_projected_distances(&points_mean_b));
    // Relative difference in point variance
    predictors
        .columns_mut(24, 3)
        .copy_from(&iter_relative_difference(
            &points_variance_a,
            &points_variance_b,
        ));
    // Covariance differences between point variances
    predictors
        .columns_mut(27, 3)
        .copy_from(&covariance_differences(
            &points_variance_a,
            &points_variance_b,
            &points_covariance_ab,
        ));
    // Relative difference in omnivariance of points
    predictors
        .column_mut(30)
        .copy_from(&omnivariance_differences(
            &points_variance_a,
            &points_variance_b,
        ));
    // Entropy of points
    predictors
        .column_mut(31)
        .copy_from(&entropy(&points_variance_a, &points_variance_b));
    // Relative difference in anisotropy, planarity, and linearity of points
    predictors
        .column_mut(32)
        .copy_from(&anisotropy_planarity_linearity(
            &points_variance_a,
            &points_variance_b,
            0,
            2,
        ));
    predictors
        .column_mut(33)
        .copy_from(&anisotropy_planarity_linearity(
            &points_variance_a,
            &points_variance_b,
            1,
            2,
        ));
    predictors
        .column_mut(34)
        .copy_from(&anisotropy_planarity_linearity(
            &points_variance_a,
            &points_variance_b,
            0,
            1,
        ));
    // Relative difference in surface variation of points
    predictors
        .column_mut(35)
        .copy_from(&surface_variation(&points_variance_a, &points_variance_b));
    // Relative difference in sphericity of points
    predictors
        .column_mut(36)
        .copy_from(&sphericity(&points_variance_a, &points_variance_b));
    // Angular similarity between distorted and reference planes
    predictors
        .column_mut(37)
        .copy_from(&angular_similarity(&points_eigenvectors_b_y));
    // Parallelity of distorted planes
    predictors
        .column_mut(38)
        .copy_from(&parallelity(&points_eigenvectors_b_x, 0));
    predictors
        .column_mut(39)
        .copy_from(&parallelity(&points_eigenvectors_b_z, 2));
    predictors
}
