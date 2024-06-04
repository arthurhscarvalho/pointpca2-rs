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

pub fn preprocess_point_cloud<'a>(
    points: &'a na::DMatrix<f64>,
    colors: &'a na::DMatrix<u8>,
) -> (&'a na::DMatrix<f64>, &'a na::DMatrix<u8>) {
    let (points_result, colors_result) = duplicate_merging(points, colors);
    return (points_result, colors_result);
}
