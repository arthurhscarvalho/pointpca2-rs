use libm::acos;
use na::{MatrixView, MatrixViewMut};
use num_traits::Pow;
use std::f64::{consts::PI, EPSILON};

pub fn relative_difference(x: f64, y: f64) -> f64 {
    1. - (x - y).abs() / (x.abs() + y.abs() + EPSILON)
}

pub fn iter_relative_difference<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
    target: &'a mut MatrixViewMut<f64, T, T>,
) {
    let nrows = x.nrows();
    let ncols = x.ncols();
    for i in 0..nrows {
        for j in 0..ncols {
            target[(i, j)] = relative_difference(x[(i, j)], y[(i, j)]);
        }
    }
}

pub fn covariance_differences<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
    z: &'a MatrixView<f64, T, T>,
    target: &'a mut MatrixViewMut<f64, T, T>,
) {
    let nrows = x.nrows();
    let ncols = x.ncols();
    for i in 0..nrows {
        for j in 0..ncols {
            let sqrt_product_diff = (x[(i, j)].sqrt() * y[(i, j)].sqrt() - z[(i, j)]).abs();
            let sqrt_product_norm = x[(i, j)].sqrt() * y[(i, j)].sqrt() + EPSILON;
            target[(i, j)] = sqrt_product_diff / sqrt_product_norm;
        }
    }
}

pub fn textural_variance_sum<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
    target: &'a mut MatrixViewMut<f64, T, T>,
) {
    let nrows = x.nrows();
    for i in 0..nrows {
        let x_sum: f64 = x.row(i).sum();
        let y_sum: f64 = y.row(i).sum();
        target[(i, 0)] = relative_difference(x_sum, y_sum);
    }
}

pub fn omnivariance_differences<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
    target: &'a mut MatrixViewMut<f64, T, T>,
) {
    let nrows = x.nrows();
    for i in 0..nrows {
        let x_prod: f64 = x.row(i).product();
        let y_prod: f64 = y.row(i).product();
        target[(i, 0)] = relative_difference(x_prod.cbrt(), y_prod.cbrt());
    }
}

pub fn entropy<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
    target: &'a mut MatrixViewMut<f64, T, T>,
) {
    let nrows = x.nrows();
    let ncols = x.ncols();
    for i in 0..nrows {
        let mut x_entropy: f64 = 0.;
        let mut y_entropy: f64 = 0.;
        for j in 0..ncols {
            x_entropy += x[(i, j)] * (x[(i, j)] + EPSILON).ln();
            y_entropy += y[(i, j)] * (y[(i, j)] + EPSILON).ln();
        }
        target[(i, 0)] = relative_difference(x_entropy, y_entropy);
    }
}

pub fn euclidean_distances<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
    target: &'a mut MatrixViewMut<f64, T, T>,
) {
    let nrows = x.nrows();
    let ncols = x.ncols();
    for i in 0..nrows {
        let mut dists: f64 = 0.;
        for j in 0..ncols {
            dists += (x[(i, j)] - y[(i, j)]).pow(2);
        }
        target[(i, 0)] = dists.sqrt();
    }
}

pub fn vector_projected_distances<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
    col: usize,
    target: &'a mut MatrixViewMut<f64, T, T>,
) {
    let nrows = x.nrows();
    for i in 0..nrows {
        target[(i, 0)] = (x[(i, col)] - y[(i, col)]).abs();
    }
}

pub fn point_projected_distances<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    target: &'a mut MatrixViewMut<f64, T, T>,
) {
    let nrows = x.nrows();
    for i in 0..nrows {
        target[(i, 0)] = x[(i, 1)].abs();
        target[(i, 1)] = x[(i, 2)].abs();
    }
}

pub fn point_to_centroid_distances<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    target: &'a mut MatrixViewMut<f64, T, T>,
) {
    let nrows = x.nrows();
    for i in 0..nrows {
        let mut dists: f64 = 0.;
        for j in 0..x.ncols() {
            dists += x[(i, j)].pow(2);
        }
        target[(i, 0)] = dists.sqrt();
    }
}

pub fn anisotropy_planarity_linearity<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
    col1: usize,
    col2: usize,
    target: &'a mut MatrixViewMut<f64, T, T>,
) {
    let nrows = x.nrows();
    for i in 0..nrows {
        let x_diff = (x[(i, col1)] - x[(i, col2)]) / (x[(i, 0)] + EPSILON);
        let y_diff = (y[(i, col1)] - y[(i, col2)]) / (y[(i, 0)] + EPSILON);
        target[(i, 0)] = relative_difference(x_diff, y_diff);
    }
}

pub fn surface_variation<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
    target: &'a mut MatrixViewMut<f64, T, T>,
) {
    let nrows = x.nrows();
    for i in 0..nrows {
        let x_sum: f64 = x.row(i).sum();
        let y_sum: f64 = y.row(i).sum();
        target[(i, 0)] =
            relative_difference(x[(i, 2)] / (x_sum + EPSILON), y[(i, 2)] / (y_sum + EPSILON));
    }
}

pub fn sphericity<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    y: &'a MatrixView<f64, T, T>,
    target: &'a mut MatrixViewMut<f64, T, T>,
) {
    let nrows = x.nrows();
    for i in 0..nrows {
        target[(i, 0)] = relative_difference(
            x[(i, 2)] / (x[(i, 0)] + EPSILON),
            y[(i, 2)] / (y[(i, 0)] + EPSILON),
        );
    }
}

pub fn angular_similarity<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    target: &'a mut MatrixViewMut<f64, T, T>,
) {
    let nrows = x.nrows();
    for i in 0..nrows {
        let numerator = x[(i, 1)];
        let (mut a, mut b, mut c) = (x[(i, 0)], x[(i, 1)], x[(i, 2)]);
        (a, b, c) = (a.pow(2), b.pow(2), c.pow(2));
        let denominator: f64 = (a + b + c).sqrt() + EPSILON;
        target[(i, 0)] = 1. - 2. * acos((numerator / denominator).abs()) / PI;
    }
}

pub fn parallelity<'a, T: na::Dim>(
    x: &'a MatrixView<f64, T, T>,
    col: usize,
    target: &'a mut MatrixViewMut<f64, T, T>,
) {
    for (i, num) in x.column(col).iter().enumerate() {
        target[i] = 1. - *num;
    }
}
