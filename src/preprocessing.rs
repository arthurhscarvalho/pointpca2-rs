use crate::utils;
use na::DMatrix;
use ordered_float::OrderedFloat;
use std::{collections::HashMap, vec};

fn vec_mean(
    vector: Vec<(OrderedFloat<f64>, OrderedFloat<f64>, OrderedFloat<f64>)>,
) -> (f64, f64, f64) {
    let vec_len = vector.len() as f64;
    let mut vec_sum = (0., 0., 0.);
    for (a, b, c) in vector {
        vec_sum.0 += utils::from_ordered(a);
        vec_sum.1 += utils::from_ordered(b);
        vec_sum.2 += utils::from_ordered(c);
    }
    let mean = (
        vec_sum.0 / vec_len,
        vec_sum.1 / vec_len,
        vec_sum.2 / vec_len,
    );
    mean
}

pub fn duplicate_merging<'a>(
    points: &'a na::DMatrix<f64>,
    colors: &'a na::DMatrix<u8>,
) -> (na::DMatrix<f64>, na::DMatrix<u8>) {
    let mut points_map: HashMap<
        (OrderedFloat<f64>, OrderedFloat<f64>, OrderedFloat<f64>),
        Vec<(OrderedFloat<f64>, OrderedFloat<f64>, OrderedFloat<f64>)>,
    > = HashMap::new();
    for i in 0..points.nrows() {
        let point = points.row(i);
        let point = (
            utils::to_ordered(point[0]),
            utils::to_ordered(point[1]),
            utils::to_ordered(point[2]),
        );
        let color = colors.row(i);
        let color = (
            utils::to_ordered(color[0] as f64),
            utils::to_ordered(color[1] as f64),
            utils::to_ordered(color[2] as f64),
        );
        if !points_map.contains_key(&point) {
            points_map.insert(point, vec![color]);
        } else if let Some(colors) = points_map.get_mut(&point) {
            colors.push(color);
        };
    }
    let nrows = points_map.len();
    let mut points_result = DMatrix::zeros(nrows, 3);
    let mut colors_result = DMatrix::zeros(nrows, 3);
    for (i, &key) in points_map.keys().enumerate() {
        let point = (
            utils::from_ordered(key.0),
            utils::from_ordered(key.1),
            utils::from_ordered(key.2),
        );
        if let Some(colors) = points_map.get(&key) {
            let colors_mean = vec_mean(colors.clone());
            points_result[(i, 0)] = point.0;
            points_result[(i, 1)] = point.1;
            points_result[(i, 2)] = point.2;
            colors_result[(i, 0)] = colors_mean.0 as u8;
            colors_result[(i, 1)] = colors_mean.1 as u8;
            colors_result[(i, 2)] = colors_mean.2 as u8;
        } else {
            panic!("Error during duplicate points merging.")
        }
    }
    return (points_result, colors_result);
}

fn rgb_to_yuv<'a>(rgb: &'a na::DMatrix<u8>) -> na::DMatrix<u8> {
    let (rows, cols) = rgb.shape();
    assert!(
        cols == 3,
        "Input matrix must have 3 columns representing RGB values"
    );
    let rgb_f64 = rgb.map(|val| val as f64);
    let r = rgb_f64.column(0);
    let g = rgb_f64.column(1);
    let b = rgb_f64.column(2);
    let c = na::DMatrix::from_row_slice(
        3,
        3,
        &[
            0.2126, 0.7152, 0.0722, -0.1146, -0.3854, 0.5000, 0.5000, -0.4542, -0.0468,
        ],
    );
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
) -> (na::DMatrix<f64>, na::DMatrix<u8>) {
    let (points_merged, colors_merged) = duplicate_merging(points, colors);
    let colors_yuv = rgb_to_yuv(&colors_merged);
    return (points_merged, colors_yuv);
}
