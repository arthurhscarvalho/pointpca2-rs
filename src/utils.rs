extern crate ordered_float;
use self::ordered_float::OrderedFloat;
use na::{Const, DMatrix, Dyn, Matrix, RowVector3, Scalar, VecStorage};
use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::ops::AddAssign;

pub fn to_ordered(num: f64) -> OrderedFloat<f64> {
    OrderedFloat(num)
}

pub fn from_ordered(num: OrderedFloat<f64>) -> f64 {
    num.into()
}

pub fn print_if_verbose<'a>(string: &'a str, verbose: &'a bool) {
    if *verbose {
        println!("{}", string);
    }
}

pub fn dmatrix_to_vec_f64<T>(matrix: DMatrix<T>) -> Vec<[f64; 3]>
where
    T: Send + Sync + AsPrimitive<f64>,
{
    matrix
        .row_iter()
        .collect::<Vec<_>>()
        .par_iter_mut()
        .map(|p| [p[0].as_(), p[1].as_(), p[2].as_()])
        .collect()
}

pub fn slice_from_knn_indices<'a>(
    points: &'a Vec<[f64; 3]>,
    colors: &'a Vec<[f64; 3]>,
    knn_indices: &'a DMatrix<usize>,
    search_size: usize,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let sl_knn_indices = knn_indices.columns(0, search_size);
    let nrows = search_size;
    let ncols = points[0].len();
    let mut selected_points = DMatrix::zeros(nrows, ncols);
    let mut selected_colors = DMatrix::zeros(nrows, ncols);
    for (i, j) in sl_knn_indices.iter().enumerate() {
        selected_points
            .row_mut(i)
            .copy_from(&RowVector3::from_row_slice(&points[*j]));
        selected_colors
            .row_mut(i)
            .copy_from(&RowVector3::from_row_slice(&colors[*j]));
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
    matrix: &'a DMatrix<f64>,
    row_vec: &'a Matrix<f64, Const<1>, Dyn, VecStorage<f64, Const<1>, Dyn>>,
) -> DMatrix<f64> {
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
