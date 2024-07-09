# PointPCA2 - Rust
#### An implementation of PointPCA2 in Rust

This project features a Rust implementation of PointPCA2, designed for faster feature computation and improved RAM management. A statistical test will be conducted to compare results from the original and Rust implementations, ensuring the validity and reliability of the adaptation.

## Roadmap
- [x] Implementation: a fully working version of PointPCA2, written entirely in Rust
- [x] Testing: extensively test the project on entire datasets 
- [x] Statistical comparison: conduct a t-test to statistically compare the features generated from the orignal and Rust implementations

## Setup

### Prerequisites
- rustc >= 1.77.2

### Build
Simply clone this repository and run ```cargo run -r```. It it not recommended to run this project without the ```-r``` flag as the computation will be very slow for entire point clouds.

## Usage
Please refer to the *main.rs* file as it contains an example of the usage. Please keep in mind that the function for reading point clouds is **very** experimental.

```rust
fn main() {
    let search_size = 81;
    println!("Reading ply");
    let (points_a, colors_a) = ply_manager::read_ply_as_matrix("<path-to-reference>");
    let (points_b, colors_b) = ply_manager::read_ply_as_matrix("<path-to-test>");
    println!("Preprocessing");
    let (points_a, colors_a) = preprocessing::preprocess_point_cloud(&points_a, &colors_a);
    let (points_b, colors_b) = preprocessing::preprocess_point_cloud(&points_b, &colors_b);
    println!("Performing knn search");
    let knn_indices_a = knn_search::knn_search(&points_a, &points_a, search_size);
    let knn_indices_b = knn_search::knn_search(&points_b, &points_a, search_size);
    println!("Computing local features");
    let local_features = features::compute_features(
        &points_a,
        &colors_a,
        &points_b,
        &colors_b,
        &knn_indices_a,
        &knn_indices_b,
        search_size,
    );
    println!("Computing predictors");
    let predictors_result = predictors::compute_predictors(&local_features);
    println!("Pooling predictors");
    let pooled_predictors = pooling::mean_pooling(&predictors_result);
    println!("Predictors:");
    for col in pooled_predictors.iter() {
        print!("{:.4}  ", *col);
    }
    println!("");
}
```

## Validity
A statistical test was conducted to validate the implementation's accuracy and reliability. Feature sets were generated from each implementation using the entire dataset. These features were split into train and test sets using GroupKFold. The training features were then fitted to the subjective scores from the dataset. Pearson and Spearman correlation coefficients were calculated, and a paired t-test was performed on these correlations.

<details>
    <summary>Spoiler</summary>
<br>

| Model                         | p-value (Pearson) | p_value ≤ 0.01 (Pearson) | p-value (Spearman) | p_value ≤ 0.01 (Spearman) |
|-------------------------------|-------------------|--------------------------|--------------------|---------------------------|
| AdaBoostRegressor             | 0.8284            | False                    | 0.4353             | False                     |
| BaggingRegressor              | 0.7707            | False                    | 0.4670             | False                     |
| BayesianRidge                 | 0.1170            | False                    | 0.2571             | False                     |
| DecisionTreeRegressor         | 0.2939            | False                    | 0.5070             | False                     |
| DummyRegressor                | 0.3005            | False                    | 0.1369             | False                     |
| ElasticNet                    | 0.2886            | False                    | 0.9511             | False                     |
| ElasticNetCV                  | 0.4883            | False                    | 0.4001             | False                     |
| ExtraTreeRegressor            | 0.4713            | False                    | 0.5298             | False                     |
| ExtraTreesRegressor           | 0.4079            | False                    | 0.1260             | False                     |
| GammaRegressor                | 0.1311            | False                    | 0.1949             | False                     |
| GaussianProcessRegressor      | 0.9629            | False                    | 0.1338             | False                     |
| GradientBoostingRegressor     | 0.1676            | False                    | 0.3176             | False                     |
| HistGradientBoostingRegressor | 0.3126            | False                    | 0.1461             | False                     |
| HuberRegressor                | 0.9787            | False                    | 0.6584             | False                     |
| KNeighborsRegressor           | 0.1921            | False                    | 0.1921             | False                     |
| KernelRidge                   | 0.0204            | False                    | 0.4711             | False                     |
| LGBMRegressor                 | 0.4033            | False                    | 0.7499             | False                     |
| Lars                          | 0.9952            | False                    | 0.9284             | False                     |
| LarsCV                        | 0.3800            | False                    | 0.3711             | False                     |
| Lasso                         | 0.7884            | False                    | 0.2102             | False                     |
| LassoCV                       | 0.0927            | False                    | 0.9883             | False                     |
| LassoLars                     | 0.7883            | False                    | 0.3739             | False                     |
| LassoLarsCV                   | 0.0655            | False                    | 0.3445             | False                     |
| LassoLarsIC                   | 0.4121            | False                    | 0.6207             | False                     |
| LinearRegression              | 0.1350            | False                    | 0.2236             | False                     |
| LinearSVR                     | 0.6661            | False                    | 0.9499             | False                     |
| MLPRegressor                  | 0.1959            | False                    | 0.2018             | False                     |
| NuSVR                         | 0.3044            | False                    | 0.4866             | False                     |
| OrthogonalMatchingPursuit     | 0.7381            | False                    | 0.1737             | False                     |
| OrthogonalMatchingPursuitCV   | 0.8318            | False                    | 0.2025             | False                     |
| PassiveAggressiveRegressor    | 0.4464            | False                    | 0.3981             | False                     |
| PoissonRegressor              | 0.4355            | False                    | 0.1636             | False                     |
| RANSACRegressor               | 0.7823            | False                    | 0.8516             | False                     |
| RandomForestRegressor         | 0.6716            | False                    | 0.9506             | False                     |
| Ridge                         | 0.0204            | False                    | 0.4711             | False                     |
| RidgeCV                       | 0.5322            | False                    | 0.5073             | False                     |
| SGDRegressor                  | 0.1674            | False                    | 0.5050             | False                     |
| SVR                           | 0.0987            | False                    | 0.0799             | False                     |
| TransformedTargetRegressor    | 0.1350            | False                    | 0.2236             | False                     |
| TweedieRegressor              | 0.9226            | False                    | 0.9126             | False                     |
| XGBRegressor                  | 0.7213            | False                    | 0.6575             | False                     |

*P-values rounded to 4 decimal places to improve readability.*

</details>

## Results
Here we compare the results of this implementation with the original.

<details>
    <summary>Spoiler</summary>
Firstly, we can compare the average time taken for the computation of features for an entire dataset.
<br><br>

| Implementation | Average time taken (seconds) |
|----------------|------------------------------|
| MATLAB         | 140.1177001453079            |
| pointpca2-rs   | 60.807681022019224           |

We can also calculate the absolute differences between corresponding features and then determine the maximum absolute difference. Additionally, we can compute the standard deviation of these absolute differences and find the highest standard deviation among them.

| Maximum absolute difference | Maximum standard deviation |
|-----------------------------|----------------------------|
| 0.10911592087732802         | 0.026780771726352532       |

Finally, we compare the correlation indices, splitting the dataset and fitting the features similarly to the previous section.

<img src="https://i.imgur.com/RrskslL.png">
</details>

## Contributing
Feel free to open issues to this project, any kind of contributions are greatly appreciated.

## References
- MATLAB implementation of PointPCA 2:

  [cwi-dis/pointpca2](https://github.com/cwi-dis/pointpca2/) - 2023 Grand Challenge on Objective Quality Metrics for Volumetric Contents
- Point clouds used in the comparisons and examples:

  E. Alexiou, I. Viola, T. M. Borges, T. A. Fonseca, R. L. De Queiroz, and T. Ebrahimi, “A comprehensive study of the rate-distortion performance in mpeg point cloud compression,” APSIPA Transactions on Signal and Information Processing, vol. 8, 2019

## License
GNU GENERAL PUBLIC LICENSE<br>
Version 2, June 1991

---

> GitHub [@akaTsunemori](https://github.com/akaTsunemori)

