use na::DMatrix;

fn svd_sign_correction(
    mut u: DMatrix<f64>,
    mut v: DMatrix<f64>,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let nrows = v.nrows();
    let ncols = v.ncols();
    let mut max_abs_rows = vec![0; nrows];
    for i in 0..nrows {
        let mut max_abs_val = f64::NEG_INFINITY;
        for j in 0..ncols {
            let abs_val = v[(i, j)].abs();
            if abs_val > max_abs_val {
                max_abs_val = abs_val;
                max_abs_rows[i] = j;
            }
        }
    }
    let signs: Vec<f64> = max_abs_rows
        .iter()
        .enumerate()
        .map(|(i, &j)| v[(i, j)].signum())
        .collect();
    for i in 0..nrows {
        for j in 0..ncols {
            v[(i, j)] *= signs[i];
            u[(j, i)] *= signs[i];
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
                cov += (x[(k, i)] - means[i]) * (x[(k, j)] - means[j]);
            }
            covariance_matrix[(i, j)] = cov / (nrows as f64 - 1.);
        }
    }
    covariance_matrix
}

pub fn compute_eigenvectors<'a>(matrix: &'a DMatrix<f64>) -> DMatrix<f64> {
    let covariance_matrix = compute_covariance_matrix(matrix);
    let eigenvectors = covariance_matrix.svd(true, true);
    let u = eigenvectors.u.unwrap();
    let v_t = eigenvectors.v_t.unwrap();
    let (u_corrected, _) = svd_sign_correction(u, v_t);
    u_corrected
}