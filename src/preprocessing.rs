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

fn rgb_to_yuv(rgb: [u8; 3]) -> [u8; 3] {
    let r = rgb[0] as f32;
    let g = rgb[1] as f32;
    let b = rgb[2] as f32;
    let y = (0.2126 * r + 0.2126 * g + 0.0722 * b) + 0.;
    let u = (-0.1146 * r + -0.3854 * g + 0.5000 * b) + 128.;
    let v = (0.5000 * r + -0.4542 * g + -0.0468 * b) + 128.;
    [
        y.round().clamp(0.0, 255.0) as u8,
        u.round().clamp(0.0, 255.0) as u8,
        v.round().clamp(0.0, 255.0) as u8,
    ]
}

fn mean_colors(colors: Vec<[u8; 3]>) -> [u8; 3] {
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

pub fn preprocess_point_cloud(
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
        let unique_point = from_ordered_point(ordered_point);
        let mean_color = mean_colors(colors);
        let color_yuv = rgb_to_yuv(mean_color);
        points_result.push(unique_point);
        colors_result.push(color_yuv);
    }
    (points_result, colors_result)
}
