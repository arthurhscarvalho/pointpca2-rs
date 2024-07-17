use crate::utils;
use na::DMatrix;

fn svd_sign_correction(mut u: DMatrix<f64>, mut v: DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
    let nrows = v.nrows();
    let mut sign;
    for i in 0..nrows {
        sign = v
            .row(i)
            .iter()
            .max_by(|&&a, &&b| a.abs().total_cmp(&b.abs()))
            .unwrap()
            .signum();
        v.row_mut(i).scale_mut(sign);
        u.column_mut(i).scale_mut(sign);
    }
    (u, v)
}

fn compute_population_covariance(x: &DMatrix<f64>) -> DMatrix<f64> {
    let nrows = x.nrows() as f64;
    let means = x.row_mean();
    let centered = utils::subtract_row_from_matrix(&x, &means);
    let covariance_matrix = (&centered.transpose() * &centered) / nrows;
    covariance_matrix
}

pub fn compute_eigenvectors<'a>(matrix: &'a DMatrix<f64>) -> DMatrix<f64> {
    let covariance_matrix = compute_population_covariance(matrix);
    let eigenvectors = covariance_matrix.svd(true, true);
    let u = eigenvectors.u.unwrap();
    let v_t = eigenvectors.v_t.unwrap();
    let (u_corrected, _) = svd_sign_correction(u, v_t);
    u_corrected
}
