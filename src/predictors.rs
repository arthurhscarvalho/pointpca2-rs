use crate::spatial_metrics;
use na::DMatrix;

pub fn compute_predictors(local_features: DMatrix<f64>) -> DMatrix<f64> {
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
    let nrows = local_features.nrows();
    let ncols = 40;
    let mut predictors = DMatrix::zeros(nrows, ncols);
    /*
        Textural predictors
    */
    // Relative differences in mean color values
    spatial_metrics::iter_relative_difference(
        &colors_mean_a,
        &colors_mean_b,
        &mut predictors.columns_mut(0, 3),
    );
    // Relative differences in color variances
    spatial_metrics::iter_relative_difference(
        &colors_variance_a,
        &colors_variance_b,
        &mut predictors.columns_mut(3, 3),
    );
    // Covariance differences between color variances
    spatial_metrics::covariance_differences(
        &colors_variance_a,
        &colors_variance_b,
        &colors_covariance_ab,
        &mut predictors.columns_mut(6, 3),
    );
    // Sum of variances of textures
    spatial_metrics::textural_variance_sum(
        &colors_variance_a,
        &colors_variance_b,
        &mut predictors.columns_mut(9, 1),
    );
    // Relative differences in omnivariance of textures
    spatial_metrics::omnivariance_differences(
        &colors_variance_a,
        &colors_variance_b,
        &mut predictors.columns_mut(10, 1),
    );
    // Entropy of textures
    spatial_metrics::entropy(
        &colors_variance_a,
        &colors_variance_b,
        &mut predictors.columns_mut(11, 1),
    );
    /*
        Geometric predictors
    */
    // Euclidean distances between distorted and reference points (error vector)
    spatial_metrics::euclidean_distances(
        &projection_a_to_a,
        &projection_b_to_a,
        &mut predictors.columns_mut(12, 1),
    );
    // Projected distances of vectors between distorted and reference points from reference planes
    for i in 0..3 {
        spatial_metrics::vector_projected_distances(
            &projection_a_to_a,
            &projection_b_to_a,
            i,
            &mut predictors.columns_mut(13 + i, 1),
        );
    }
    // Projected distances of reference points from reference planes
    spatial_metrics::point_projected_distances(
        &projection_a_to_a,
        &mut predictors.columns_mut(16, 2),
    );
    // Euclidean distances between distorted points and reference centroids
    spatial_metrics::point_to_centroid_distances(
        &projection_b_to_a,
        &mut predictors.columns_mut(18, 1),
    );
    // Projected distances of distorted points from reference planes
    spatial_metrics::point_projected_distances(
        &projection_b_to_a,
        &mut predictors.columns_mut(19, 2),
    );
    // Euclidean distances between distorted centroids and reference centroids
    spatial_metrics::point_to_centroid_distances(
        &points_mean_b,
        &mut predictors.columns_mut(21, 1),
    );
    // Projected distances of distorted centroids from reference planes
    spatial_metrics::point_projected_distances(&points_mean_b, &mut predictors.columns_mut(22, 2));
    // Relative differences in points variances
    spatial_metrics::iter_relative_difference(
        &points_variance_a,
        &points_variance_b,
        &mut predictors.columns_mut(24, 3),
    );
    // Covariance differences between points variances
    spatial_metrics::covariance_differences(
        &points_variance_a,
        &points_variance_b,
        &points_covariance_ab,
        &mut predictors.columns_mut(27, 3),
    );
    // Relative difference in omnivariance of points
    spatial_metrics::omnivariance_differences(
        &points_variance_a,
        &points_variance_b,
        &mut predictors.columns_mut(30, 1),
    );
    // Entropy of points
    spatial_metrics::entropy(
        &points_variance_a,
        &points_variance_b,
        &mut predictors.columns_mut(31, 1),
    );
    // Relative differences in anisotropy, planarity, and linearity of points
    for (i, col1, col2) in [(0, 0, 2), (1, 1, 2), (2, 0, 1)] {
        spatial_metrics::anisotropy_planarity_linearity(
            &points_variance_a,
            &points_variance_b,
            col1,
            col2,
            &mut predictors.columns_mut(32 + i, 1),
        );
    }
    // Relative differences in surface variation of points
    spatial_metrics::surface_variation(
        &points_variance_a,
        &points_variance_b,
        &mut predictors.columns_mut(35, 1),
    );
    // Relative differences in sphericity of points
    spatial_metrics::sphericity(
        &points_variance_a,
        &points_variance_b,
        &mut predictors.columns_mut(36, 1),
    );
    // Angular similarity between distorted and reference planes
    spatial_metrics::angular_similarity(&eigenvectors_b_y, &mut predictors.columns_mut(37, 1));
    // Parallelity of distorted planes
    spatial_metrics::parallelity(&eigenvectors_b_x, 0, &mut predictors.columns_mut(38, 1));
    spatial_metrics::parallelity(&eigenvectors_b_z, 2, &mut predictors.columns_mut(39, 1));
    predictors
}
