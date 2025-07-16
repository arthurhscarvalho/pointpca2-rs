use na::{Const, DMatrix, Dyn, Matrix, Scalar, VecStorage};
use std::ops::AddAssign;

pub fn print_if_verbose<'a>(string: &'a str, verbose: &'a bool) {
    if *verbose {
        println!("{}", string);
    }
}

pub fn slice_from_knn_indices<'a>(
    points: &'a Vec<[f32; 3]>,
    colors: &'a Vec<[u8; 3]>,
    knn_indices: &'a Vec<usize>,
    search_size: usize,
) -> (DMatrix<f32>, DMatrix<f32>) {
    let nrows = search_size;
    let ncols = points[0].len();
    let mut selected_points = DMatrix::zeros(nrows, ncols);
    let mut selected_colors = DMatrix::zeros(nrows, ncols);
    for (i, &nbr_idx) in knn_indices.iter().enumerate() {
        selected_points.row_mut(i).copy_from_slice(&points[nbr_idx]);
        selected_colors
            .row_mut(i)
            .copy_from_slice(&colors[nbr_idx].map(|c| c as f32));
    }
    (selected_points, selected_colors)
}

pub fn concatenate_columns<'a, T>(mat1: &'a DMatrix<T>, mat2: &'a DMatrix<T>) -> DMatrix<T>
where
    T: Scalar + Copy + AddAssign + num_traits::identities::Zero,
{
    assert_eq!(
        mat1.nrows(),
        mat2.nrows(),
        "Matrices must have the same number of rows for concatenation."
    );
    let mut result = DMatrix::zeros(mat1.nrows(), mat1.ncols() + mat2.ncols());
    result
        .view_mut((0, 0), (mat1.nrows(), mat1.ncols()))
        .copy_from(&mat1);
    result
        .view_mut((0, mat1.ncols()), (mat2.nrows(), mat2.ncols()))
        .copy_from(&mat2);
    result
}

pub fn subtract_row_from_matrix<'a>(
    matrix: &'a DMatrix<f32>,
    row_vec: &'a Matrix<f32, Const<1>, Dyn, VecStorage<f32, Const<1>, Dyn>>,
) -> DMatrix<f32> {
    assert_eq!(row_vec.nrows(), 1, "row_vec must be a single-row vector.");
    assert_eq!(
        matrix.ncols(),
        row_vec.ncols(),
        "Arguments must have the same number of columns."
    );
    let mut new_matrix = matrix.clone();
    new_matrix.row_iter_mut().for_each(|mut row| row -= row_vec);
    new_matrix
}
