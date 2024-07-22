# PointPCA2 - Rust
#### An implementation of PointPCA2 in Rust

This project features a Rust implementation of PointPCA2, designed for faster feature computation and improved RAM management. A statistical test will be conducted to compare results from the original and Rust implementations, ensuring the validity and reliability of the adaptation.

## Roadmap
- [x] Implementation: a fully working version of PointPCA2, written entirely in Rust
- [x] Testing: extensively test the project on entire datasets
- [x] Statistical comparison: compare the correlation between features generated from the original and Rust implementations

## Setup

### Prerequisites
- rustc >= 1.77.2

### Build
Simply clone this repository and run ```cargo run -r```. It it not recommended to run this project without the ```-r``` flag as the computation will be very slow for entire point clouds.

## Usage
Please refer to the *main.rs* file as it contains an example of the usage. Please keep in mind that the function for reading point clouds is **very** experimental.

```rust
use pointpca2_rs;
use pointpca2_rs::ply_manager;

fn main() {
    let search_size = 81;
    let verbose = true;
    println!("Reading ply");
    let (points_a, colors_a) = ply_manager::read_ply_as_matrix("<path-to-reference>");
    let (points_b, colors_b) = ply_manager::read_ply_as_matrix("<path-to-test>");
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

## Validation
To validate the accuracy and reliability of the Rust implementation, we conducted a feature-wise correlation analysis between the reference (original) implementation and the test (pointpca2-rs) implementation. Initially, the pointpca2 features were extracted from a comprehensive dataset (refer to references for dataset details) using both implementations. This procedure yielded a table comprising 40 columns, each representing a distinct feature, and 232 rows, corresponding to the number of point clouds in the dataset. Subsequently, the Pearson Correlation Coefficient (PCC), the Spearman Ranking Order Correlation Coefficient (SROCC), the Mean Absolute Error (MAE), and the Mean Squared Error (MSE) were computed for each column, resulting in a correlation table with 40 rows, where each row pertains to the correlations of a specific feature, and five columns, indicating the feature index, PCC, SROCC, MAE, and MSE respectively.

<details>
    <summary>Spoiler</summary>
<br>

| Feature | PLCC   | SROCC  | MAE    | MSE    |
|---------|--------|--------|--------|--------|
| 1       | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2       | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 3       | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 4       | 1.0000 | 1.0000 | 0.0001 | 0.0000 |
| 5       | 1.0000 | 1.0000 | 0.0001 | 0.0000 |
| 6       | 1.0000 | 1.0000 | 0.0001 | 0.0000 |
| 7       | 0.9968 | 0.9999 | 0.0016 | 0.0000 |
| 8       | 0.9998 | 1.0000 | 0.0005 | 0.0000 |
| 9       | 0.9999 | 1.0000 | 0.0004 | 0.0000 |
| 10      | 1.0000 | 1.0000 | 0.0001 | 0.0000 |
| 11      | 1.0000 | 1.0000 | 0.0001 | 0.0000 |
| 12      | 1.0000 | 1.0000 | 0.0001 | 0.0000 |
| 13      | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 14      | 0.9984 | 0.9998 | 0.0070 | 0.0001 |
| 15      | 0.9993 | 0.9998 | 0.0068 | 0.0002 |
| 16      | 0.9928 | 0.9989 | 0.0134 | 0.0005 |
| 17      | 1.0000 | 0.9985 | 0.0065 | 0.0000 |
| 18      | 1.0000 | 1.0000 | 0.0003 | 0.0000 |
| 19      | 0.9991 | 0.9999 | 0.0123 | 0.0003 |
| 20      | 0.9973 | 0.9998 | 0.0066 | 0.0001 |
| 21      | 0.9618 | 0.9972 | 0.0199 | 0.0012 |
| 22      | 0.9999 | 1.0000 | 0.0026 | 0.0000 |
| 23      | 0.9997 | 1.0000 | 0.0021 | 0.0000 |
| 24      | 0.9997 | 1.0000 | 0.0007 | 0.0000 |
| 25      | 1.0000 | 1.0000 | 0.0002 | 0.0000 |
| 26      | 1.0000 | 1.0000 | 0.0002 | 0.0000 |
| 27      | 1.0000 | 1.0000 | 0.0001 | 0.0000 |
| 28      | 0.8321 | 1.0000 | 0.0013 | 0.0000 |
| 29      | 0.9367 | 1.0000 | 0.0013 | 0.0000 |
| 30      | 0.9953 | 1.0000 | 0.0016 | 0.0000 |
| 31      | 1.0000 | 1.0000 | 0.0001 | 0.0000 |
| 32      | 1.0000 | 1.0000 | 0.0002 | 0.0000 |
| 33      | 0.9999 | 1.0000 | 0.0001 | 0.0000 |
| 34      | 0.9998 | 1.0000 | 0.0004 | 0.0000 |
| 35      | 0.9987 | 1.0000 | 0.0015 | 0.0000 |
| 36      | 1.0000 | 1.0000 | 0.0002 | 0.0000 |
| 37      | 0.9999 | 1.0000 | 0.0003 | 0.0000 |
| 38      | 0.9987 | 0.9999 | 0.0010 | 0.0000 |
| 39      | 0.9981 | 0.9999 | 0.0014 | 0.0000 |
| 40      | 0.9647 | 0.9961 | 0.0032 | 0.0000 |

*Values rounded to 4 decimal places for better readability.*

</details>

## Results
Here, we compare the results of this implementation with the original. The benchmarks were done on an i5-10400F with 2x8 GB RAM @ 2666 MHz.

<details>
    <summary>Spoiler</summary>
<br>

Firstly, we can compare the average time taken for the computation of features for an entire dataset.
<br>

| Implementation | Average time taken (seconds) |
|----------------|------------------------------|
| MATLAB         | 140.1177001453079            |
| pointpca2-rs   | 6.681233939425699            |

We can also calculate the absolute differences between corresponding features and then determine the maximum absolute difference. Additionally, we can compute the standard deviation of these absolute differences and find the highest standard deviation among them.

| Maximum absolute difference | Maximum standard deviation |
|-----------------------------|----------------------------|
| 0.11058533454477848         | 0.027662635634742926       |

Feature sets were derived from each implementation utilizing the entire dataset (refer to references). These features were partitioned into training and testing sets using Leave One Group Out. LazyPredict was employed to fit the training features to the subjective scores from the dataset using all available regressors. Pearson and Spearman correlation coefficients were computed to compare the predicted (test) scores and the subjective (reference) scores, and a comparative plot was generated to visualize the results.

<img src="https://i.imgur.com/oaknzk7.png">
</details>

## Contributing
Feel free to open issues to this project, any kind of contributions are greatly appreciated.

## References
- MATLAB implementation of PointPCA 2:

  [cwi-dis/pointpca2](https://github.com/cwi-dis/pointpca2/) - 2023 Grand Challenge on Objective Quality Metrics for Volumetric Contents

- Point clouds dataset used in the validation and results sections:

  E. Alexiou, I. Viola, T. M. Borges, T. A. Fonseca, R. L. De Queiroz, and T. Ebrahimi, “A comprehensive study of the rate-distortion performance in mpeg point cloud compression,” APSIPA Transactions on Signal and Information Processing, vol. 8, 2019

## License
MIT License

---

> GitHub [@akaTsunemori](https://github.com/akaTsunemori)
