use na::DMatrix;

use crate::utils;

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
            let abs_val = u[(i, j)].abs();
            if abs_val > max_abs_val {
                max_abs_val = abs_val;
                max_abs_cols[j] = i;
            }
        }
    }
    let signs: Vec<f64> = max_abs_cols
        .iter()
        .enumerate()
        .map(|(j, &idx)| u[(idx, j)].signum())
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

fn compute_covariance_matrix<'a>(data: &'a DMatrix<f64>) -> DMatrix<f64> {
    let n = data.ncols();
    let m = data.column_mean();
    let mut covariance_matrix = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut cov = 0.0;
            for k in 0..data.nrows() {
                cov += (data[(k, i)] - m[i]) * (data[(k, j)] - m[j]);
            }
            covariance_matrix[(i, j)] = cov / (data.nrows() as f64 - 1.0);
            covariance_matrix[(j, i)] = covariance_matrix[(i, j)];
        }
    }
    return covariance_matrix;
}

fn compute_eigenvectors<'a>(matrix: &'a DMatrix<f64>) -> DMatrix<f64> {
    let covariance_matrix = compute_covariance_matrix(matrix);
    let eigenvectors = covariance_matrix.svd(true, true);
    let u = eigenvectors.u.unwrap();
    // let s = eigenvectors.singular_values;
    let v_t = eigenvectors.v_t.unwrap();
    let (u_corrected, _) = svd_sign_correction(u, v_t);
    return u_corrected;
}

fn slice_from_knn_indices<'a>(
    points: &'a na::DMatrix<f64>,
    colors: &'a na::DMatrix<u8>,
    knn_indices: &'a na::DMatrix<i64>,
    knn_row: usize,
    search_size: usize,
) -> (DMatrix<f64>, DMatrix<u8>) {
    let knn_indices_row = knn_indices.row(knn_row);
    let sl_knn_indices = knn_indices_row.columns(0, search_size);
    let mut selected_points = Vec::new();
    let mut selected_colors = Vec::new();
    for j in sl_knn_indices.iter() {
        selected_points.push(points.row(*j as usize));
        selected_colors.push(colors.row(*j as usize));
    }
    let sl_points = DMatrix::from_rows(&selected_points);
    let sl_colors = DMatrix::from_rows(&selected_colors);
    return (sl_points, sl_colors);
}

pub fn compute_features<'a>(
    points_a: &'a na::DMatrix<f64>,
    colors_a: &'a na::DMatrix<u8>,
    points_b: &'a na::DMatrix<f64>,
    colors_b: &'a na::DMatrix<u8>,
    knn_indices_a: &'a na::DMatrix<i64>,
    knn_indices_b: &'a na::DMatrix<i64>,
) -> DMatrix<f64> {
    // let search_size = 81; // Temporary
    let search_size = 9; // Temporary
    let nrows = points_a.shape().0;
    let ncols = 42;
    let mut local_features = DMatrix::zeros(nrows, ncols);
    local_features.fill(f64::NAN);
    for i in 0..nrows {
        let (sl_points_a, sl_colors_a) =
            slice_from_knn_indices(points_a, colors_a, knn_indices_a, i, search_size);
        let (sl_points_b, sl_colors_b) =
            slice_from_knn_indices(points_b, colors_b, knn_indices_b, i, search_size);
        let eigenvectors_a = compute_eigenvectors(&sl_points_a);
        let projection_a_to_a =
            utils::subtract_row_from_matrix(&sl_points_a, &sl_points_a.row_mean())
                * eigenvectors_a.clone();
        let projection_b_to_a =
            utils::subtract_row_from_matrix(&sl_points_b, &sl_points_a.row_mean())
                * eigenvectors_a.clone();
        let sl_colors_a_f64 = sl_colors_a.map(|x| x as f64);
        let sl_colors_b_f64 = sl_colors_b.map(|x| x as f64);
        let mean_a = utils::concatenate_columns(&projection_a_to_a, &sl_colors_a_f64).row_mean();
        let mean_b = utils::concatenate_columns(&projection_b_to_a, &sl_colors_b_f64).row_mean();
        let proj_colors_a_concat = utils::concatenate_columns(&projection_a_to_a, &sl_colors_a_f64);
        let proj_colors_b_concat = utils::concatenate_columns(&projection_b_to_a, &sl_colors_b_f64);
        let mean_deviation_a =
            utils::subtract_row_from_matrix(&proj_colors_a_concat, &mean_a.clone());
        let mean_deviation_b =
            utils::subtract_row_from_matrix(&proj_colors_b_concat, &mean_b.clone());
        let variance_a = mean_deviation_a.map(|x| x * x).row_mean();
        let variance_b = mean_deviation_b.map(|x| x * x).row_mean();
        let mut covariance_ab = mean_deviation_a.clone();
        for j in 0..covariance_ab.nrows() {
            for k in 0..covariance_ab.ncols() {
                covariance_ab[(j, k)] *= mean_deviation_b[(j, k)];
            }
        }
        let covariance_ab = covariance_ab.row_mean();
        let eigenvectors_b = compute_eigenvectors(&projection_b_to_a);
        // // Update local features
        local_features
            .row_mut(i)
            .view_mut((0, 0), (1, 3))
            .copy_from(&projection_a_to_a.row(0));
        local_features
            .row_mut(i)
            .view_mut((0, 3), (1, 3))
            .copy_from(&projection_b_to_a.row(0));
        local_features
            .row_mut(i)
            .view_mut((0, 6), (1, 3))
            .copy_from(&mean_a.view((0, 3), (1, 3)));
        local_features
            .row_mut(i)
            .view_mut((0, 9), (1, 6))
            .copy_from(&mean_b);
        local_features
            .row_mut(i)
            .view_mut((0, 15), (1, 6))
            .copy_from(&variance_a);
        local_features
            .row_mut(i)
            .view_mut((0, 21), (1, 6))
            .copy_from(&variance_b);
        local_features
            .row_mut(i)
            .view_mut((0, 27), (1, 6))
            .copy_from(&covariance_ab);
        local_features
            .row_mut(i)
            .view_mut((0, 33), (1, 3))
            .copy_from(&eigenvectors_b.view((0, 0), (1, 3)));
        local_features
            .row_mut(i)
            .view_mut((0, 36), (1, 3))
            .copy_from(&eigenvectors_b.view((1, 0), (1, 3)));
        local_features
            .row_mut(i)
            .view_mut((0, 39), (1, 3))
            .copy_from(&eigenvectors_b.view((2, 0), (1, 3)));
    }
    return local_features;
}
