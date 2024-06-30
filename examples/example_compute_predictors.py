import open3d as o3d
import numpy as np
import pointpca2_rs


PC_REF_PATH = "/home/arthurc/APSIPA___M-PCCD/references/redandblack_vox10_1550.ply"
pc_ref = o3d.io.read_point_cloud(PC_REF_PATH)
points_a, colors_a = np.asarray(pc_ref.points), np.asarray(pc_ref.colors)
PC_TEST_PATH = "/home/arthurc/APSIPA___M-PCCD/PVS/tmc13_redandblack_vox10_1550_dec_geom01_text01_trisoup-predlift.ply"
pc_test = o3d.io.read_point_cloud(PC_TEST_PATH)
points_b, colors_b = np.asarray(pc_test.points), np.asarray(pc_test.colors)

predictors = pointpca2_rs.pointpca2(
    points_a, colors_a, points_b, colors_b, search_size=81, verbose=False
)
print(*[f'{i:.4f}' for i in predictors])
