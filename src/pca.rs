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
    // Sort indices by eigenvalues in descending order (largest first)
    let mut indices = [0, 1, 2];
    let eigenvalues = &eigen.eigenvalues;
    // Manual sorting for 3x3 matrix - faster than Vec allocation and sort
    if eigenvalues[indices[0]] < eigenvalues[indices[1]] {
        indices.swap(0, 1);
    }
    if eigenvalues[indices[1]] < eigenvalues[indices[2]] {
        indices.swap(1, 2);
    }
    if eigenvalues[indices[0]] < eigenvalues[indices[1]] {
        indices.swap(0, 1);
    }
    let sorted_eigenvectors = DMatrix::from_columns(
        &indices
            .iter()
            .map(|&i| eigen.eigenvectors.column(i))
            .collect::<Vec<_>>(),
    );
    // Sign correction for eigenvectors
    let u_corrected = eigen_sign_correction(sorted_eigenvectors);
    u_corrected
}

pub fn compute_pca(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let covariance_matrix = compute_covariance_matrix(matrix, false);
    let pca_matrix = compute_eigenvectors(covariance_matrix);
    pca_matrix
}
