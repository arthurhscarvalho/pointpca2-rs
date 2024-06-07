use kiddo::float::{distance::SquaredEuclidean, kdtree::KdTree};
use na::DMatrix;

pub fn knn_search<'a>(
    xa: &'a DMatrix<f64>,
    xb: &'a DMatrix<f64>,
    search_size: usize,
) -> DMatrix<usize> {
    let mut query = [0., 0., 0.];
    let mut kdtree: KdTree<f64, usize, 3, 1024, u32> = KdTree::with_capacity(xa.nrows());
    let mut knn_indices = DMatrix::zeros(xb.nrows(), search_size);
    for i in 0..xa.nrows() {
        query[0] = xa[(i, 0)];
        query[1] = xa[(i, 1)];
        query[2] = xa[(i, 2)];
        kdtree.add(&(query.clone()), i);
    }
    for i in 0..xb.nrows() {
        query[0] = xb[(i, 0)];
        query[1] = xb[(i, 1)];
        query[2] = xb[(i, 2)];
        let neighbors = kdtree.nearest_n::<SquaredEuclidean>(&query, search_size);
        let indices = neighbors
            .into_iter()
            .map(|nbr| nbr.item)
            .collect::<Vec<usize>>();
        let indices = DMatrix::from_row_slice(1, search_size, &indices);
        knn_indices.view_mut((i, 0), (1, search_size)).copy_from(&indices);
    }
    return knn_indices;
}
