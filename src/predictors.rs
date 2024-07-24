use crate::spatial_metrics;
use na::{DMatrix, Matrix1xX};

pub fn compute_predictors(local_features: DMatrix<f64>) -> Matrix1xX<f64> {
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
    let mut predictors = Matrix1xX::zeros(40);
    /*
        Textural predictors
    */
    // Relative differences in mean color values
    predictors.columns_mut(0, 3).copy_from(
        &spatial_metrics::iter_relative_difference(&colors_mean_a, &colors_mean_b).row_mean(),
    );
    // Relative differences in color variances
    predictors.columns_mut(3, 3).copy_from(
        &spatial_metrics::iter_relative_difference(&colors_variance_a, &colors_variance_b)
            .row_mean(),
    );
    // Covariance differences between color variances
    predictors.columns_mut(6, 3).copy_from(
        &spatial_metrics::covariance_differences(
            &colors_variance_a,
            &colors_variance_b,
            &colors_covariance_ab,
        )
        .row_mean(),
    );
    // Sum of variances of textures
    predictors.column_mut(9).copy_from(
        &spatial_metrics::textural_variance_sum(&colors_variance_a, &colors_variance_b).row_mean(),
    );
    // Relative differences in omnivariance of textures
    predictors.column_mut(10).copy_from(
        &spatial_metrics::omnivariance_differences(&colors_variance_a, &colors_variance_b)
            .row_mean(),
    );
    // Entropy of textures
    predictors
        .column_mut(11)
        .copy_from(&spatial_metrics::entropy(&colors_variance_a, &colors_variance_b).row_mean());
    /*
        Geometric predictors
    */
    // Euclidean distances between distorted and reference points (error vector)
    predictors.column_mut(12).copy_from(
        &spatial_metrics::euclidean_distances(&projection_a_to_a, &projection_b_to_a).row_mean(),
    );
    // Projected distances of vectors between distorted and reference points from reference planes
    predictors.column_mut(13).copy_from(
        &spatial_metrics::vector_projected_distances(&projection_a_to_a, &projection_b_to_a, 0)
            .row_mean(),
    );
    predictors.column_mut(14).copy_from(
        &spatial_metrics::vector_projected_distances(&projection_a_to_a, &projection_b_to_a, 1)
            .row_mean(),
    );
    predictors.column_mut(15).copy_from(
        &spatial_metrics::vector_projected_distances(&projection_a_to_a, &projection_b_to_a, 2)
            .row_mean(),
    );
    // Projected distances of reference points from reference planes
    predictors
        .columns_mut(16, 2)
        .copy_from(&spatial_metrics::point_projected_distances(&projection_a_to_a).row_mean());
    // Euclidean distances between distorted points and reference centroids
    predictors
        .column_mut(18)
        .copy_from(&spatial_metrics::point_to_centroid_distances(&projection_b_to_a).row_mean());
    // Projected distances of distorted points from reference planes
    predictors
        .columns_mut(19, 2)
        .copy_from(&spatial_metrics::point_projected_distances(&projection_b_to_a).row_mean());
    // Euclidean distances between distorted centroids and reference centroids
    predictors
        .column_mut(21)
        .copy_from(&spatial_metrics::point_to_centroid_distances(&points_mean_b).row_mean());
    // Projected distances of distorted centroids from reference planes
    predictors
        .columns_mut(22, 2)
        .copy_from(&spatial_metrics::point_projected_distances(&points_mean_b).row_mean());
    // Relative differences in points variances
    predictors.columns_mut(24, 3).copy_from(
        &spatial_metrics::iter_relative_difference(&points_variance_a, &points_variance_b)
            .row_mean(),
    );
    // Covariance differences between points variances
    predictors.columns_mut(27, 3).copy_from(
        &spatial_metrics::covariance_differences(
            &points_variance_a,
            &points_variance_b,
            &points_covariance_ab,
        )
        .row_mean(),
    );
    // Relative difference in omnivariance of points
    predictors.column_mut(30).copy_from(
        &spatial_metrics::omnivariance_differences(&points_variance_a, &points_variance_b)
            .row_mean(),
    );
    // Entropy of points
    predictors
        .column_mut(31)
        .copy_from(&spatial_metrics::entropy(&points_variance_a, &points_variance_b).row_mean());
    // Relative differences in anisotropy, planarity, and linearity of points
    predictors.column_mut(32).copy_from(
        &spatial_metrics::anisotropy_planarity_linearity(
            &points_variance_a,
            &points_variance_b,
            0,
            2,
        )
        .row_mean(),
    );
    predictors.column_mut(33).copy_from(
        &spatial_metrics::anisotropy_planarity_linearity(
            &points_variance_a,
            &points_variance_b,
            1,
            2,
        )
        .row_mean(),
    );
    predictors.column_mut(34).copy_from(
        &spatial_metrics::anisotropy_planarity_linearity(
            &points_variance_a,
            &points_variance_b,
            0,
            1,
        )
        .row_mean(),
    );
    // Relative differences in surface variation of points
    predictors.column_mut(35).copy_from(
        &spatial_metrics::surface_variation(&points_variance_a, &points_variance_b).row_mean(),
    );
    // Relative differences in sphericity of points
    predictors
        .column_mut(36)
        .copy_from(&spatial_metrics::sphericity(&points_variance_a, &points_variance_b).row_mean());
    // Angular similarity between distorted and reference planes
    predictors
        .column_mut(37)
        .copy_from(&spatial_metrics::angular_similarity(&eigenvectors_b_y).row_mean());
    // Parallelity of distorted planes
    predictors
        .column_mut(38)
        .copy_from(&spatial_metrics::parallelity(&eigenvectors_b_x, 0).row_mean());
    predictors
        .column_mut(39)
        .copy_from(&spatial_metrics::parallelity(&eigenvectors_b_z, 2).row_mean());
    predictors
}
