import numpy as np
from .pointpca2_rs import pointpca2 as __pointpca2_internal


def __preprocess_point_cloud(points, colors):
    if type(points) != np.ndarray:
        points = np.array(points, dtype=np.double)
    if type(colors) != np.ndarray:
        colors = np.array(colors, dtype=np.uint8)
    if points.shape != colors.shape:
        raise Exception("Points and colors must have the same shape.")
    if colors.max() <= 1 and colors.min() >= 0:
        colors = (colors * 255).astype(np.uint8)
    return points, colors


def pointpca2(points_a, colors_a, points_b, colors_b, search_size=81, verbose=False):
    points_a, colors_a = __preprocess_point_cloud(points_a, colors_a)
    points_b, colors_b = __preprocess_point_cloud(points_b, colors_b)
    predictors = __pointpca2_internal(
        points_a, colors_a, points_b, colors_b, search_size, verbose
    )
    return predictors
