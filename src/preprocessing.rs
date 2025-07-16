use ordered_float::OrderedFloat;
use std::collections::BTreeMap;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct OrderedPoint(OrderedFloat<f32>, OrderedFloat<f32>, OrderedFloat<f32>);

fn to_ordered_point(num: [f32; 3]) -> OrderedPoint {
    OrderedPoint(
        OrderedFloat(num[0]),
        OrderedFloat(num[1]),
        OrderedFloat(num[2]),
    )
}

fn from_ordered_point(num: OrderedPoint) -> [f32; 3] {
    [num.0.into(), num.1.into(), num.2.into()]
}

fn colors_mean(colors: Vec<[u8; 3]>) -> [u8; 3] {
    let vec_len = colors.len() as f32;
    let mut vec_sum = (0, 0, 0);
    for color in colors {
        vec_sum.0 += color[0];
        vec_sum.1 += color[1];
        vec_sum.2 += color[2];
    }
    let mean = (
        vec_sum.0 as f32 / vec_len,
        vec_sum.1 as f32 / vec_len,
        vec_sum.2 as f32 / vec_len,
    );
    let rounded_mean = [
        mean.0.round() as u8,
        mean.1.round() as u8,
        mean.2.round() as u8,
    ];
    rounded_mean
}

pub fn duplicate_merging(
    points: Vec<[f32; 3]>,
    colors: Vec<[u8; 3]>,
) -> (Vec<[f32; 3]>, Vec<[u8; 3]>) {
    let mut points_map: BTreeMap<OrderedPoint, Vec<[u8; 3]>> = BTreeMap::new();
    for i in 0..points.len() {
        let point = to_ordered_point(points[i]);
        points_map
            .entry(point)
            .or_insert_with(Vec::new)
            .push(colors[i]);
    }
    let nrows = points_map.len();
    let mut points_result = Vec::with_capacity(nrows);
    let mut colors_result = Vec::with_capacity(nrows);
    for (ordered_point, colors) in points_map {
        let point = from_ordered_point(ordered_point);
        let colors_mean = colors_mean(colors);
        points_result.push(point);
        colors_result.push(colors_mean);
    }
    (points_result, colors_result)
}

fn rgb_to_yuv(rgb: Vec<[u8; 3]>) -> Vec<[u8; 3]> {
    rgb.into_iter()
        .map(|color| {
            let r = color[0] as f32;
            let g = color[1] as f32;
            let b = color[2] as f32;
            let y = (0.2126 * r + 0.2126 * g + 0.0722 * b) + 0.;
            let u = (-0.1146 * r + -0.3854 * g + 0.5000 * b) + 128.;
            let v = (0.5000 * r + -0.4542 * g + -0.0468 * b) + 128.;
            [
                y.round().clamp(0.0, 255.0) as u8,
                u.round().clamp(0.0, 255.0) as u8,
                v.round().clamp(0.0, 255.0) as u8,
            ]
        })
        .collect()
}

pub fn preprocess_point_cloud(
    points: Vec<[f32; 3]>,
    colors: Vec<[u8; 3]>,
) -> (Vec<[f32; 3]>, Vec<[u8; 3]>) {
    let (points, colors) = duplicate_merging(points, colors);
    let colors = rgb_to_yuv(colors);
    (points, colors)
}
