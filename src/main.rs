extern crate nalgebra as na;
extern crate ordered_float;
extern crate kiddo;

use knn_search::knn_search;
use ordered_float::OrderedFloat;
use std::collections::HashSet;

mod features;
mod predictors;
mod preprocessing;
mod utils;
mod knn_search;

fn main() {
    let points_a = na::DMatrix::from_row_slice(
        10,
        3,
        &[
            0., 67., 209., 0., 68., 208., 0., 68., 209., 0., 68., 210., 0., 68., 211., 0., 69.,
            208., 0., 69., 209., 0., 69., 211., 0., 69., 212., 0., 70., 207.,
        ],
    );
    let colors_a = na::DMatrix::from_row_slice(
        10,
        3,
        &[
            185, 110, 139, 183, 103, 139, 169, 106, 139, 191, 103, 139, 173, 101, 139, 183, 104,
            138, 170, 105, 139, 174, 102, 139, 208, 105, 138, 165, 106, 139,
        ],
    );
    let points_b = na::DMatrix::from_row_slice(
        10,
        3,
        &[
            0., 64., 207., 0., 65., 207., 0., 66., 207., 0., 67., 207., 0., 68., 206., 0., 68.,
            207., 0., 69., 206., 0., 69., 207., 0., 70., 206., 0., 70., 207.,
        ],
    );
    let colors_b = na::DMatrix::from_row_slice(
        10,
        3,
        &[
            181, 110, 137, 183, 110, 137, 185, 110, 137, 187, 110, 137, 188, 110, 137, 189, 110,
            137, 187, 110, 137, 188, 110, 137, 187, 110, 137, 187, 110, 137,
        ],
    );
    let knn_indices_a = na::DMatrix::from_row_slice(
        10,
        9,
        &[
            0, 2, 1, 3, 6, 5, 4, 7, 8, 1, 2, 5, 6, 0, 3, 9, 4, 7, 2, 6, 0, 1, 3, 5, 4, 7, 9, 3, 4,
            2, 6, 0, 7, 1, 8, 5, 4, 3, 7, 8, 2, 6, 0, 1, 5, 5, 1, 6, 2, 9, 3, 0, 7, 4, 6, 2, 5, 1,
            3, 7, 0, 9, 4, 7, 8, 4, 3, 6, 2, 0, 5, 1, 8, 7, 4, 3, 6, 2, 0, 5, 1, 9, 5, 1, 6, 2, 3,
            0, 7, 4,
        ],
    );
    let knn_indices_b = na::DMatrix::from_row_slice(
        10,
        9,
        &[
            3, 5, 2, 1, 7, 4, 6, 9, 0, 5, 7, 3, 4, 2, 6, 9, 8, 1, 5, 3, 7, 2, 9, 4, 6, 8, 1, 5, 3,
            7, 2, 9, 4, 6, 1, 8, 5, 7, 3, 2, 9, 4, 1, 6, 8, 7, 9, 5, 6, 4, 3, 8, 2, 1, 7, 5, 9, 3,
            6, 4, 8, 2, 1, 7, 5, 9, 3, 2, 6, 4, 8, 1, 7, 5, 9, 3, 2, 6, 4, 8, 1, 9, 7, 8, 6, 5, 4,
            3, 2, 1,
        ],
    );
    let local_features = features::compute_features(&points_a, &colors_a, &points_b, &colors_b, &knn_indices_a, &knn_indices_b);
    // // println!("{}", local_features);
    // for row in local_features.row_iter() {
    //     // Create a string for each row with formatted elements
    //     let formatted_row: Vec<String> = row.iter()
    //                                         .map(|&x| format!("{:.2}", x))
    //                                         .collect();
    //     // Print the formatted row
    //     println!("{}", formatted_row.join(" "));
    // }

    let predictors_result = predictors::compute_predictors(&local_features);
    for row in predictors_result.row_iter() {
        for col in row.iter() {
            print!("{:.2} ", col);
        }
        println!("");
    }

    // let knn_indices = knn_search(points_a, points_b);
    // println!("{}", knn_indices);

    // let num: f64 = 13.55667;
    // let num_ordered = utils::to_ordered(num);
    // let mut set: HashSet<OrderedFloat<f64>> = HashSet::new();
    // set.insert(num_ordered);
    // println!("HashSet containing an ordered float:");
    // println!("{:?}", set);
    // println!();
    // println!("Ordered float converted back to native Rust");
    // println!("{}", utils::from_ordered(num_ordered));
    // println!();
    // println!("Usage of DMatrix:");
    // let points = na::DMatrix::from_row_slice(5, 3, &[
    //     1.0, 2.0, 3.3,
    //     1.0, 2.0, 3.3,
    //     7.0, 8.0, 9.0,
    //     10.3, 11.0, 12.0,
    //     13.0, 14.0, 15.0,
    // ]);
    // let colors = na::DMatrix::from_row_slice(5, 3, &[
    //     4, 5, 6,
    //     4, 5, 6,
    //     7, 8, 9,
    //     10, 11, 12,
    //     13, 14, 15,
    // ]);
    // let (points_result, colors_result) = preprocessing::preprocess_point_cloud(&points, &colors);
    // for i in 0..points_result.nrows() {
    //     for j in 0..points_result.ncols() {
    //         print!("{:<8}", points_result[(i, j)]);
    //     }
    //     println!();
    // }
    // println!();
    // for i in 0..colors_result.nrows() {
    //     for j in 0..colors_result.ncols() {
    //         print!("{:<8}", colors_result[(i, j)]);
    //     }
    //     println!();
    // }
}
