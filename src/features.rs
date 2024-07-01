use crate::utils;
use na::DMatrix;

fn svd_sign_correction(
    mut u: na::DMatrix<f64>,
    mut v: na::DMatrix<f64>,
) -> (na::DMatrix<f64>, na::DMatrix<f64>) {
    let ncols = u.ncols();
    let nrows = u.nrows();
    let mut max_abs_cols = vec![0; ncols];
    for j in 0..ncols {
        let mut max_abs_val = 0.0;
        for i in 0..nrows {
            let abs_val = v[(j, i)].abs();
            if abs_val > max_abs_val {
                max_abs_val = abs_val;
                max_abs_cols[j] = i;
            }
        }
    }
    let signs: Vec<f64> = max_abs_cols
        .iter()
        .enumerate()
        .map(|(j, &idx)| v[(j, idx)].signum())
        .collect();
    for j in 0..ncols {
        for i in 0..nrows {
            u[(i, j)] *= signs[j];
        }
    }
    for (j, &sign) in signs.iter().enumerate() {
        for i in 0..v.nrows() {
            v[(i, j)] *= sign;
        }
    }
    (u, v)
}

fn compute_covariance_matrix<'a>(x: &'a DMatrix<f64>) -> DMatrix<f64> {
    let nrows = x.nrows();
    let ncols = x.ncols();
    let means = x.row_mean();
    let mut covariance_matrix = DMatrix::<f64>::zeros(ncols, ncols);
    for i in 0..ncols {
        for j in 0..ncols {
            let mut cov = 0.;
            for k in 0..nrows {
                cov += (x[(k, i)] - means[j]) * (x[(k, j)] - means[j]);
            }
            covariance_matrix[(i, j)] = cov / (nrows as f64 - 1.);
        }
    }
    covariance_matrix
}

fn compute_eigenvectors<'a>(matrix: &'a DMatrix<f64>) -> DMatrix<f64> {
    let covariance_matrix = compute_covariance_matrix(matrix);
    let eigenvectors = covariance_matrix.svd(true, true);
    let u = eigenvectors.u.unwrap();
    let v_t = eigenvectors.v_t.unwrap();
    let (u_corrected, _) = svd_sign_correction(u, v_t);
    u_corrected
}

fn slice_from_knn_indices<'a>(
    points: &'a na::DMatrix<f64>,
    colors: &'a na::DMatrix<u8>,
    knn_indices: &'a na::DMatrix<usize>,
    knn_row: usize,
    search_size: usize,
) -> (DMatrix<f64>, DMatrix<u8>) {
    let knn_indices_row = knn_indices.row(knn_row);
    let sl_knn_indices = knn_indices_row.columns(0, search_size);
    let nrows = search_size;
    let ncols = points.ncols();
    let mut selected_points = DMatrix::zeros(nrows, ncols);
    let mut selected_colors = DMatrix::zeros(nrows, ncols);
    for (i, j) in sl_knn_indices.iter().enumerate() {
        selected_points.row_mut(i).copy_from(&points.row(*j));
        selected_colors.row_mut(i).copy_from(&colors.row(*j));
    }
    (selected_points, selected_colors)
}

pub fn compute_features<'a>(
    points_a: &'a na::DMatrix<f64>,
    colors_a: &'a na::DMatrix<u8>,
    points_b: &'a na::DMatrix<f64>,
    colors_b: &'a na::DMatrix<u8>,
    knn_indices_a: &'a na::DMatrix<usize>,
    knn_indices_b: &'a na::DMatrix<usize>,
    search_size: usize,
) -> DMatrix<f64> {
    let nrows = points_a.shape().0;
    let ncols = 42;
    let mut local_features = DMatrix::zeros(nrows, ncols);
    local_features.fill(f64::NAN);
    for i in 0..nrows {
        // Slice points and colors from their respective knn indices
        let (sl_points_a, sl_colors_a) =
            slice_from_knn_indices(points_a, colors_a, knn_indices_a, i, search_size);
        let (sl_points_b, sl_colors_b) =
            slice_from_knn_indices(points_b, colors_b, knn_indices_b, i, search_size);
        // Principal components of reference data (new orthonormal basis)
        let eigenvectors_a = compute_eigenvectors(&sl_points_a);
        // Project reference and distorted data onto the new orthonormal basis
        let projection_a_to_a =
            utils::subtract_row_from_matrix(&sl_points_a, &sl_points_a.row_mean())
                * &eigenvectors_a;
        let projection_b_to_a =
            utils::subtract_row_from_matrix(&sl_points_b, &sl_points_a.row_mean())
                * &eigenvectors_a;
        // Mean values for projected geometric data and texture data
        let sl_colors_a_f64 = sl_colors_a.map(|x| x as f64);
        let sl_colors_b_f64 = sl_colors_b.map(|x| x as f64);
        let mean_a = utils::concatenate_columns(&projection_a_to_a, &sl_colors_a_f64).row_mean();
        let mean_b = utils::concatenate_columns(&projection_b_to_a, &sl_colors_b_f64).row_mean();
        let proj_colors_a_concat = utils::concatenate_columns(&projection_a_to_a, &sl_colors_a_f64);
        let proj_colors_b_concat = utils::concatenate_columns(&projection_b_to_a, &sl_colors_b_f64);
        // Deviation from mean
        let mean_deviation_a = utils::subtract_row_from_matrix(&proj_colors_a_concat, &mean_a);
        let mean_deviation_b = utils::subtract_row_from_matrix(&proj_colors_b_concat, &mean_b);
        // Variances and covariance
        let variance_a = mean_deviation_a.map(|x| x * x).row_mean();
        let variance_b = mean_deviation_b.map(|x| x * x).row_mean();
        let mut covariance_ab = mean_deviation_a.clone();
        for j in 0..covariance_ab.nrows() {
            for k in 0..covariance_ab.ncols() {
                covariance_ab[(j, k)] *= mean_deviation_b[(j, k)];
            }
        }
        let covariance_ab = covariance_ab.row_mean();
        // Principal components of projected distorted data
        let eigenvectors_b = compute_eigenvectors(&projection_b_to_a);
        // Update local features
        local_features
            .view_mut((i, 0), (1, 3))
            .copy_from(&projection_a_to_a.row(0));
        local_features
            .view_mut((i, 3), (1, 3))
            .copy_from(&projection_b_to_a.row(0));
        local_features
            .view_mut((i, 6), (1, 3))
            .copy_from(&mean_a.view((0, 3), (1, 3)));
        local_features.view_mut((i, 9), (1, 6)).copy_from(&mean_b);
        local_features
            .view_mut((i, 15), (1, 6))
            .copy_from(&variance_a);
        local_features
            .view_mut((i, 21), (1, 6))
            .copy_from(&variance_b);
        local_features
            .view_mut((i, 27), (1, 6))
            .copy_from(&covariance_ab);
        local_features
            .view_mut((i, 33), (1, 3))
            .copy_from(&eigenvectors_b.view((0, 0), (1, 3)));
        local_features
            .view_mut((i, 36), (1, 3))
            .copy_from(&eigenvectors_b.view((1, 0), (1, 3)));
        local_features
            .view_mut((i, 39), (1, 3))
            .copy_from(&eigenvectors_b.view((2, 0), (1, 3)));
    }
    return local_features;
}
