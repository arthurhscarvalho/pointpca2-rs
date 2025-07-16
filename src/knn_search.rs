use kd_tree;

pub fn build_tree<'a>(points: &'a Vec<[f32; 3]>) -> kd_tree::KdIndexTree3<'a, [f32; 3]> {
    let kdtree = kd_tree::KdIndexTree3::par_build_by_ordered_float(points);
    kdtree
}

pub fn nearest_n<'a>(
    kdtree: &'a kd_tree::KdIndexTree3<[f32; 3]>,
    point: &[f32; 3],
    n: usize,
) -> Vec<usize> {
    let neighbors = kdtree.nearests(point, n);
    let indices = neighbors
        .iter()
        .map(|nbr| *nbr.item)
        .collect::<Vec<usize>>();
    indices
}
