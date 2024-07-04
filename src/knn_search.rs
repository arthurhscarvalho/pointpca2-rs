use kiddo::float::{distance::SquaredEuclidean, kdtree::KdTree};
use na::DMatrix;

pub fn knn_search<'a>(
    xa: &'a DMatrix<f64>,
    xb: &'a DMatrix<f64>,
    search_size: usize,
) -> DMatrix<usize> {
    let mut kdtree: KdTree<f64, usize, 3, 8192, u32> = KdTree::with_capacity(xa.nrows());
    let mut knn_indices = DMatrix::zeros(xb.nrows(), search_size);
    xa.row_iter().enumerate().for_each(|(idx, point)| {
        kdtree.add(&[point[0], point[1], point[2]], idx);
    });
    xb.row_iter().enumerate().for_each(|(idx, point)| {
        let neighbors =
            kdtree.nearest_n::<SquaredEuclidean>(&[point[0], point[1], point[2]], search_size);
        let indices = neighbors.iter().map(|nbr| nbr.item).collect::<Vec<usize>>();
        let indices = DMatrix::from_row_slice(1, search_size, &indices);
        knn_indices
            .view_mut((idx, 0), (1, search_size))
            .copy_from(&indices);
    });
    return knn_indices;
}
