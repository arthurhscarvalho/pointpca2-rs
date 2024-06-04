extern crate ordered_float;
use self::ordered_float::OrderedFloat;
use nalgebra::{DMatrix, Scalar, Matrix, Const, VecStorage, Dyn};
use std::ops::{AddAssign};

pub fn to_ordered(num: f64) -> OrderedFloat<f64> {
    return OrderedFloat(num);
}

pub fn from_ordered(num: OrderedFloat<f64>) -> f64 {
    return num.into();
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

pub fn subtract_row_from_matrix<'a>(matrix: &'a DMatrix<f64>, row_vec: &'a Matrix<f64, Const<1>, Dyn, VecStorage<f64, Const<1>, Dyn>>) -> DMatrix<f64>
{
    // Check if the row vector has the same number of columns as the matrix
    if row_vec.nrows() != 1 || row_vec.ncols() != matrix.ncols() {
        panic!("Row vector must have the same dimensions as the matrix columns.");
    }
    // Create a new matrix with the same dimensions as the original
    let mut new_matrix = DMatrix::zeros(matrix.nrows(), matrix.ncols());
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    // Iterate through each row of the matrix and perform element-wise subtraction
    for i in 0..nrows {
        for j in 0..ncols {
            new_matrix[(i, j)] = matrix[(i, j)] - row_vec[(0, j)];
        }
    }
    return new_matrix;
}