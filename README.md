# PointPCA2 - Rust

This project features a Rust implementation of PointPCA2 [1], designed for faster feature computation and improved RAM management.

## Setup

### Prerequisites
- rustc >= 1.77.2

### Build
Clone this repository and run `cargo build --release`. Run with `cargo run --release`.

## Usage
Please refer to the *main.rs* file as it contains an example of the usage. Please keep in mind that the function for reading point clouds is experimental.

```rust
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
```

## Contributing
Feel free to open issues to this project, any kind of contributions are greatly appreciated.

## References
[1] Zhou, X., Alexiou, E., Viola, I., & César, P. (2025). PointPCA+: A full-reference point cloud quality assessment metric with PCA‑based features. Signal Processing: Image Communication, 135, Article 117262. https://doi.org/10.1016/j.image.2025.117262

## License
MIT License

---

> GitHub [@arthurhscarvalho](https://github.com/arthurhscarvalho)
