use crate::eigenvectors;
use crate::knn_search;
use crate::utils;
use na::DMatrix;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

pub fn compute_features(
    points_a: DMatrix<f64>,
    colors_a: DMatrix<u8>,
    points_b: DMatrix<f64>,
    colors_b: DMatrix<u8>,
    search_size: usize,
) -> DMatrix<f64> {
    let points_a = utils::dmatrix_to_vec_f64(points_a);
    let colors_a = utils::dmatrix_to_vec_f64(colors_a);
    let points_b = utils::dmatrix_to_vec_f64(points_b);
    let colors_b = utils::dmatrix_to_vec_f64(colors_b);
    let kd_tree_a = knn_search::build_tree(&points_a);
    let kd_tree_b = knn_search::build_tree(&points_b);
    let nrows = points_a.len();
    let ncols = 42;
    let mut local_features = DMatrix::zeros(nrows, ncols);
    local_features
        .row_iter_mut()
        .collect::<Vec<_>>()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, row)| {
            let knn_indices_a = knn_search::nearest_n(&kd_tree_a, &points_a[i], search_size);
            let knn_indices_b = knn_search::nearest_n(&kd_tree_b, &points_a[i], search_size);
            // Slice points and colors from their respective knn indices
            let (sl_points_a, sl_colors_a) =
                utils::slice_from_knn_indices(&points_a, &colors_a, &knn_indices_a, search_size);
            let (sl_points_b, sl_colors_b) =
                utils::slice_from_knn_indices(&points_b, &colors_b, &knn_indices_b, search_size);
            // Principal components of reference data (new orthonormal basis)
            let eigenvectors_a = eigenvectors::compute_eigenvectors(&sl_points_a);
            // Project reference and distorted data onto the new orthonormal basis
            let sl_points_a_mean = sl_points_a.row_mean();
            let projection_a_to_a =
                utils::subtract_row_from_matrix(&sl_points_a, &sl_points_a_mean) * &eigenvectors_a;
            let projection_b_to_a =
                utils::subtract_row_from_matrix(&sl_points_b, &sl_points_a_mean) * &eigenvectors_a;
            // Mean values for projected geometric data and texture data
            let mean_a = utils::concatenate_columns(&projection_a_to_a, &sl_colors_a).row_mean();
            let mean_b = utils::concatenate_columns(&projection_b_to_a, &sl_colors_b).row_mean();
            let proj_colors_a_concat = utils::concatenate_columns(&projection_a_to_a, &sl_colors_a);
            let proj_colors_b_concat = utils::concatenate_columns(&projection_b_to_a, &sl_colors_b);
            // Deviation from mean
            let mean_deviation_a = utils::subtract_row_from_matrix(&proj_colors_a_concat, &mean_a);
            let mean_deviation_b = utils::subtract_row_from_matrix(&proj_colors_b_concat, &mean_b);
            // Variances and covariance
            let variance_a = mean_deviation_a.map(|x| x.powi(2)).row_mean();
            let variance_b = mean_deviation_b.map(|x| x.powi(2)).row_mean();
            let covariance_ab = mean_deviation_a.component_mul(&mean_deviation_b).row_mean();
            // Principal components of projected distorted data
            let eigenvectors_b = eigenvectors::compute_eigenvectors(&projection_b_to_a).transpose();
            // Update local features
            row.columns_mut(0, 3).copy_from(&projection_a_to_a.row(0));
            row.columns_mut(3, 3).copy_from(&projection_b_to_a.row(0));
            row.columns_mut(6, 3).copy_from(&mean_a.columns(3, 3));
            row.columns_mut(9, 6).copy_from(&mean_b);
            row.columns_mut(15, 6).copy_from(&variance_a);
            row.columns_mut(21, 6).copy_from(&variance_b);
            row.columns_mut(27, 6).copy_from(&covariance_ab);
            row.columns_mut(33, 3).copy_from(&eigenvectors_b.row(0));
            row.columns_mut(36, 3).copy_from(&eigenvectors_b.row(1));
            row.columns_mut(39, 3).copy_from(&eigenvectors_b.row(2));
        });
    local_features
}
