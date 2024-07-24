use na::{DMatrix, Matrix1xX};

enum PoolingTechnique {
    MeanPooling,
    MaxPooling,
}

impl PoolingTechnique {
    fn from_str(pooling: &str) -> Option<Self> {
        match pooling {
            "mean_pooling" => Some(Self::MeanPooling),
            "max_pooling" => Some(Self::MaxPooling),
            _ => None,
        }
    }
}

pub struct Pool {
    technique: PoolingTechnique,
}

impl Pool {
    pub fn new(pooling: &str) -> Option<Self> {
        PoolingTechnique::from_str(pooling).map(|technique| Self { technique })
    }

    pub fn pool<'a>(&self, matrix: &'a DMatrix<f64>) -> Matrix1xX<f64> {
        match self.technique {
            PoolingTechnique::MeanPooling => self.mean_pooling(matrix),
            PoolingTechnique::MaxPooling => self.max_pooling(matrix),
        }
    }

    fn mean_pooling<'a>(&self, matrix: &'a DMatrix<f64>) -> Matrix1xX<f64> {
        matrix.row_mean()
    }

    fn max_pooling<'a>(&self, matrix: &'a DMatrix<f64>) -> Matrix1xX<f64> {
        (0..matrix.ncols())
            .map(|i| matrix.column(i).max())
            .collect::<Vec<_>>()
            .into()
    }
}
