use kd_tree;
use na::DMatrix;
use rayon::prelude::*;

pub fn knn_search<'a>(
    xa: &'a DMatrix<f64>,
    xb: &'a DMatrix<f64>,
    search_size: usize,
) -> DMatrix<usize> {
    let points: Vec<[f64; 3]> = xa.row_iter().map(|p| [p[0], p[1], p[2]]).collect();
    let kdtree = kd_tree::KdIndexTree3::par_build_by_ordered_float(&points);
    let mut knn_indices = DMatrix::zeros(xb.nrows(), search_size);
    let mut knn_indices_rows: Vec<_> = knn_indices.row_iter_mut().collect();
    knn_indices_rows
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, row)| {
            let point = xb.row(idx);
            let neighbors = kdtree.nearests(&[point[0], point[1], point[2]], search_size);
            let indices: Vec<usize> = neighbors.iter().map(|nbr| *nbr.item).collect();
            let indices = DMatrix::from_row_slice(1, search_size, &indices);
            row.copy_from(&indices);
        });
    knn_indices
}
