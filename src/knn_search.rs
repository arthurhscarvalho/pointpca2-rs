use kd_tree;
use na::DMatrix;

pub fn build_tree<'a>(points: &'a Vec<[f64; 3]>) -> kd_tree::KdIndexTree3<'a, [f64; 3]> {
    let kdtree = kd_tree::KdIndexTree3::par_build_by_ordered_float(points);
    kdtree
}

pub fn nearest_n<'a>(
    kdtree: &'a kd_tree::KdIndexTree3<[f64; 3]>,
    point: &[f64; 3],
    n: usize,
) -> DMatrix<usize> {
    let neighbors = kdtree.nearests(point, n);
    let indices = DMatrix::from_row_slice(
        1,
        n,
        &neighbors
            .iter()
            .map(|nbr| *nbr.item)
            .collect::<Vec<usize>>(),
    );
    indices
}
