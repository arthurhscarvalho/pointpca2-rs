use crate::utils;
use na::DMatrix;

fn compute_covariance_matrix(x: &DMatrix<f64>, unbiased: bool) -> DMatrix<f64> {
    let bias = if unbiased { 1. } else { 0. };
    let nrows = x.nrows() as f64;
    let means = x.row_mean();
    let centered = utils::subtract_row_from_matrix(&x, &means);
    let covariance_matrix = (&centered.transpose() * &centered) / (nrows - bias);
    covariance_matrix
}

fn eigen_sign_correction(mut u: DMatrix<f64>) -> DMatrix<f64> {
    let nrows = u.nrows();
    let mut sign;
    for i in 0..nrows {
        sign = u
            .column(i)
            .iter()
            .max_by(|&&a, &&b| a.abs().total_cmp(&b.abs()))
            .unwrap()
            .signum();
        u.column_mut(i).apply(|x| *x *= sign);
    }
    u
}

fn compute_eigenvectors(matrix: DMatrix<f64>) -> DMatrix<f64> {
    let eigen = matrix.symmetric_eigen();
    let mut indices: Vec<usize> = (0..eigen.eigenvalues.len()).collect();
    indices.sort_by(|&i, &j| {
        eigen.eigenvalues[j]
            .partial_cmp(&eigen.eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_eigenvectors = DMatrix::from_columns(
        &indices
            .iter()
            .map(|&i| eigen.eigenvectors.column(i).into_owned())
            .collect::<Vec<_>>(),
    );
    let u_corrected = eigen_sign_correction(sorted_eigenvectors);
    u_corrected
}

pub fn compute_pca(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let covariance_matrix = compute_covariance_matrix(matrix, false);
    let pca_matrix = compute_eigenvectors(covariance_matrix);
    pca_matrix
}
