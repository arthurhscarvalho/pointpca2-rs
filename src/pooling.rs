use na::{DMatrix, Matrix1xX};

pub fn mean_pooling<'a>(predictors: &'a DMatrix<f64>) -> Matrix1xX<f64> {
    let pooled_predictors = predictors.row_mean();
    return pooled_predictors;
}
