use na::{DMatrix, Matrix1xX};

enum PoolingTechnique {
    MeanPooling,
    MaxPooling,
    MinPooling,
    MedianPooling,
}

impl PoolingTechnique {
    fn from_str(pooling: &str) -> Option<Self> {
        match pooling {
            "mean_pooling" => Some(Self::MeanPooling),
            "max_pooling" => Some(Self::MaxPooling),
            "min_pooling" => Some(Self::MinPooling),
            "median_pooling" => Some(Self::MedianPooling),
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

    pub fn pool<'a>(&self, matrix: &'a DMatrix<f32>) -> Matrix1xX<f32> {
        match self.technique {
            PoolingTechnique::MeanPooling => self.mean_pooling(matrix),
            PoolingTechnique::MaxPooling => self.max_pooling(matrix),
            PoolingTechnique::MinPooling => self.min_pooling(matrix),
            PoolingTechnique::MedianPooling => self.median_pooling(matrix),
        }
    }

    fn mean_pooling<'a>(&self, matrix: &'a DMatrix<f32>) -> Matrix1xX<f32> {
        matrix.row_mean()
    }

    fn max_pooling<'a>(&self, matrix: &'a DMatrix<f32>) -> Matrix1xX<f32> {
        (0..matrix.ncols())
            .map(|i| matrix.column(i).max())
            .collect::<Vec<_>>()
            .into()
    }

    fn min_pooling<'a>(&self, matrix: &'a DMatrix<f32>) -> Matrix1xX<f32> {
        (0..matrix.ncols())
            .map(|i| matrix.column(i).min())
            .collect::<Vec<_>>()
            .into()
    }

    fn median_pooling<'a>(&self, matrix: &'a DMatrix<f32>) -> Matrix1xX<f32> {
        let ncols = matrix.ncols();
        let mut medians = Matrix1xX::zeros(ncols);
        for i in 0..ncols {
            let col = matrix.column(i);
            let mut col_vec = col.iter().collect::<Vec<_>>();
            col_vec.sort_by(|&a, &b| a.partial_cmp(b).unwrap());
            let len = col_vec.len();
            if len % 2 == 0 {
                medians[i] = (col_vec[len / 2] + col_vec[len / 2 - 1]) / 2.;
            } else {
                medians[i] = *col_vec[len / 2];
            }
        }
        medians
    }
}
