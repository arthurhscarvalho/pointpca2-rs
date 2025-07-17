use crate::pooling;
use crate::spatial_metrics;
use na::{DMatrix, Matrix1xX};
use rayon::prelude::*;

const PREDICTORS_DIMENSION: usize = 40;

pub fn compute_predictors(local_features: DMatrix<f32>) -> Matrix1xX<f32> {
    let projection_a_to_a = local_features.columns(0, 3);
    let projection_b_to_a = local_features.columns(3, 3);
    let colors_mean_a = local_features.columns(6, 3);
    let points_mean_b = local_features.columns(9, 3);
    let colors_mean_b = local_features.columns(12, 3);
    let points_variance_a = local_features.columns(15, 3);
    let colors_variance_a = local_features.columns(18, 3);
    let points_variance_b = local_features.columns(21, 3);
    let colors_variance_b = local_features.columns(24, 3);
    let points_covariance_ab = local_features.columns(27, 3);
    let colors_covariance_ab = local_features.columns(30, 3);
    let eigenvectors_b_x = local_features.columns(33, 3);
    let eigenvectors_b_y = local_features.columns(36, 3);
    let eigenvectors_b_z = local_features.columns(39, 3);
    let mut predictors = Matrix1xX::zeros(PREDICTORS_DIMENSION);
    let pooling = pooling::Pool::new("mean_pooling").unwrap();
    // Define all predictor computations as closures that return (start_col, num_cols, values)
    let predictor_computations: Vec<Box<dyn Fn() -> (usize, usize, Matrix1xX<f32>) + Send + Sync>> = vec![
        // Textural predictors
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::iter_relative_difference(
                &colors_mean_a,
                &colors_mean_b,
            ));
            (0, 3, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::iter_relative_difference(
                &colors_variance_a,
                &colors_variance_b,
            ));
            (3, 3, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::covariance_differences(
                &colors_variance_a,
                &colors_variance_b,
                &colors_covariance_ab,
            ));
            (6, 3, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::textural_variance_sum(
                &colors_variance_a,
                &colors_variance_b,
            ));
            (9, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::omnivariance_differences(
                &colors_variance_a,
                &colors_variance_b,
            ));
            (10, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::entropy(
                &colors_variance_a,
                &colors_variance_b,
            ));
            (11, 1, pooled)
        }),
        // Geometric predictors
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::euclidean_distances(
                &projection_a_to_a,
                &projection_b_to_a,
            ));
            (12, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::vector_projected_distances(
                &projection_a_to_a,
                &projection_b_to_a,
                0,
            ));
            (13, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::vector_projected_distances(
                &projection_a_to_a,
                &projection_b_to_a,
                1,
            ));
            (14, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::vector_projected_distances(
                &projection_a_to_a,
                &projection_b_to_a,
                2,
            ));
            (15, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::point_projected_distances(
                &projection_a_to_a,
            ));
            (16, 2, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::point_to_centroid_distances(
                &projection_b_to_a,
            ));
            (18, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::point_projected_distances(
                &projection_b_to_a,
            ));
            (19, 2, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::point_to_centroid_distances(
                &points_mean_b,
            ));
            (21, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::point_projected_distances(&points_mean_b));
            (22, 2, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::iter_relative_difference(
                &points_variance_a,
                &points_variance_b,
            ));
            (24, 3, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::covariance_differences(
                &points_variance_a,
                &points_variance_b,
                &points_covariance_ab,
            ));
            (27, 3, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::omnivariance_differences(
                &points_variance_a,
                &points_variance_b,
            ));
            (30, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::entropy(
                &points_variance_a,
                &points_variance_b,
            ));
            (31, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::anisotropy_planarity_linearity(
                &points_variance_a,
                &points_variance_b,
                0,
                2,
            ));
            (32, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::anisotropy_planarity_linearity(
                &points_variance_a,
                &points_variance_b,
                1,
                2,
            ));
            (33, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::anisotropy_planarity_linearity(
                &points_variance_a,
                &points_variance_b,
                0,
                1,
            ));
            (34, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::surface_variation(
                &points_variance_a,
                &points_variance_b,
            ));
            (35, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::sphericity(
                &points_variance_a,
                &points_variance_b,
            ));
            (36, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::angular_similarity(&eigenvectors_b_y));
            (37, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::parallelity(&eigenvectors_b_x, 0));
            (38, 1, pooled)
        }),
        Box::new(|| {
            let pooled = pooling.pool(&spatial_metrics::parallelity(&eigenvectors_b_z, 2));
            (39, 1, pooled)
        }),
    ];
    // Compute all predictors in parallel
    let results: Vec<(usize, usize, Matrix1xX<f32>)> = predictor_computations
        .into_par_iter()
        .map(|computation| computation())
        .collect();
    // Copy results back to the predictors matrix
    for (start_col, num_cols, values) in results {
        if num_cols == 1 {
            predictors.column_mut(start_col).copy_from(&values);
        } else {
            predictors
                .columns_mut(start_col, num_cols)
                .copy_from(&values);
        }
    }
    predictors
}
