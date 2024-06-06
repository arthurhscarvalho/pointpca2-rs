use kiddo::float::{distance::SquaredEuclidean, kdtree::KdTree};
use na::DMatrix;

pub fn knn_search(xa: DMatrix<f64>, xb: DMatrix<f64>) -> DMatrix<usize> {
    let search_size = 9; // Temporary
    let mut query = [0., 0., 0.];
    let mut kdtree: KdTree<f64, usize, 3, 32, u32> = KdTree::with_capacity(xa.nrows());
    let mut knn_indices: Vec<usize> = Vec::new();
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
        let mut indices = neighbors
            .into_iter()
            .map(|nbr| nbr.item as usize)
            .collect::<Vec<usize>>();
        knn_indices.append(indices.as_mut());
    }
    let knn_indices =
        DMatrix::from_row_slice(knn_indices.len() / search_size, search_size, &knn_indices);
    return knn_indices;
}
