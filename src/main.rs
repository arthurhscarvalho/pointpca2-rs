use pointpca2_rs;
use pointpca2_rs::ply_manager;

fn main() {
    let search_size = 81;
    let verbose = true;
    println!("Reading ply");
    let (points_a, colors_a) = ply_manager::read_point_cloud("<path-to-reference>");
    let (points_b, colors_b) = ply_manager::read_point_cloud("<path-to-test>");
    let pooled_predictors = pointpca2_rs::compute_pointpca2(
        points_a,
        colors_a,
        points_b,
        colors_b,
        search_size,
        verbose,
    );
    println!("Predictors:");
    for col in pooled_predictors.iter() {
        print!("{:.4}  ", *col);
    }
    println!("");
}
