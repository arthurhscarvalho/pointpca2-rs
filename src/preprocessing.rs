fn get_unique<'a>(
    points: &'a na::DMatrix<f64>,
    colors: &'a na::DMatrix<u8>,
) -> (&'a na::DMatrix<f64>, &'a na::DMatrix<u8>) {
    return (points, colors);
}

fn duplicate_merging<'a>(
    points: &'a na::DMatrix<f64>,
    colors: &'a na::DMatrix<u8>,
) -> (&'a na::DMatrix<f64>, &'a na::DMatrix<u8>) {
    let (points_result, colors_result) = get_unique(points, colors);
    return (points_result, colors_result);
}

fn rgb_to_yuv<'a>(rgb: &'a na::DMatrix<u8>) -> na::DMatrix<u8> {
    let (rows, cols) = rgb.shape();
    assert!(cols == 3, "Input matrix must have 3 columns representing RGB values");
    let rgb_f64 = rgb.map(|val| val as f64);
    let r = rgb_f64.column(0);
    let g = rgb_f64.column(1);
    let b = rgb_f64.column(2);
    let c = na::DMatrix::from_row_slice(3, 3, &[
        0.2126,  0.7152,  0.0722,
       -0.1146, -0.3854,  0.5000,
        0.5000, -0.4542, -0.0468
    ]);
    let o = na::DVector::from_row_slice(&[0.0, 128.0, 128.0]);
    let y = (c[(0, 0)] * &r + c[(0, 1)] * &g + c[(0, 2)] * &b).add_scalar(o[0]);
    let u = (c[(1, 0)] * &r + c[(1, 1)] * &g + c[(1, 2)] * &b).add_scalar(o[1]);
    let v = (c[(2, 0)] * &r + c[(2, 1)] * &g + c[(2, 2)] * &b).add_scalar(o[2]);
    let mut yuv = na::DMatrix::zeros(rows, 3);
    for i in 0..rows {
        yuv[(i, 0)] = y[i].round() as u8;
        yuv[(i, 1)] = u[i].round() as u8;
        yuv[(i, 2)] = v[i].round() as u8;
    }
    yuv
}

pub fn preprocess_point_cloud<'a>(
    points: &'a na::DMatrix<f64>,
    colors: &'a na::DMatrix<u8>,
) -> (&'a na::DMatrix<f64>, &'a na::DMatrix<u8>) {
    let (points_result, colors_result) = duplicate_merging(points, colors);
    return (points_result, colors_result);
}
