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
A statistical test was conducted to validate the implementation's accuracy and reliability. Feature sets were generated from each implementation using the entire APSIPA dataset (in references). These features were split into train and test sets using GroupKFold. The training features were then fitted to the subjective scores from the dataset. Pearson and Spearman correlation coefficients were calculated, and a paired t-test was performed on these correlations.

<details>
    <summary>Spoiler</summary>
<br>

| Model                         | p-value (Pearson) | p_value ≤ 0.01 (Pearson) | p-value (Spearman) | p_value ≤ 0.01 (Spearman) |
|-------------------------------|-------------------|--------------------------|--------------------|---------------------------|
| AdaBoostRegressor             | 0.4210            | False                    | 0.8808             | False                     |
| BaggingRegressor              | 0.7514            | False                    | 0.4436             | False                     |
| BayesianRidge                 | 0.1318            | False                    | 0.3303             | False                     |
| DecisionTreeRegressor         | 0.1976            | False                    | 0.1866             | False                     |
| DummyRegressor                | 0.6403            | False                    | 0.9046             | False                     |
| ElasticNet                    | 0.3223            | False                    | 0.3739             | False                     |
| ElasticNetCV                  | 0.3623            | False                    | 0.4909             | False                     |
| ExtraTreeRegressor            | 0.2277            | False                    | 0.3122             | False                     |
| ExtraTreesRegressor           | 0.7542            | False                    | 0.6817             | False                     |
| GammaRegressor                | 0.1196            | False                    | 0.1509             | False                     |
| GaussianProcessRegressor      | 0.9874            | False                    | 0.4165             | False                     |
| GradientBoostingRegressor     | 0.1104            | False                    | 0.1352             | False                     |
| HistGradientBoostingRegressor | 0.5579            | False                    | 0.1804             | False                     |
| HuberRegressor                | 0.5208            | False                    | 0.4355             | False                     |
| KNeighborsRegressor           | 0.1265            | False                    | 0.3974             | False                     |
| KernelRidge                   | 0.0421            | False                    | 0.3320             | False                     |
| LGBMRegressor                 | 0.8185            | False                    | 0.7405             | False                     |
| Lars                          | 0.7545            | False                    | 0.7839             | False                     |
| LarsCV                        | 0.3513            | False                    | 0.4446             | False                     |
| Lasso                         | 0.5445            | False                    | 0.2102             | False                     |
| LassoCV                       | 0.1529            | False                    | 0.8765             | False                     |
| LassoLars                     | 0.5457            | False                    | 0.3739             | False                     |
| LassoLarsCV                   | 0.0369            | False                    | 0.4682             | False                     |
| LassoLarsIC                   | 0.3857            | False                    | 0.4887             | False                     |
| LinearRegression              | 0.1857            | False                    | 0.2774             | False                     |
| LinearSVR                     | 0.2481            | False                    | 0.3309             | False                     |
| MLPRegressor                  | 0.7577            | False                    | 0.3559             | False                     |
| NuSVR                         | 0.2313            | False                    | 0.2034             | False                     |
| OrthogonalMatchingPursuit     | 0.7615            | False                    | 0.2895             | False                     |
| OrthogonalMatchingPursuitCV   | 0.8359            | False                    | 0.2587             | False                     |
| PassiveAggressiveRegressor    | 0.3338            | False                    | 0.2266             | False                     |
| PoissonRegressor              | 0.3404            | False                    | 0.5764             | False                     |
| RANSACRegressor               | 0.7373            | False                    | 0.4991             | False                     |
| RandomForestRegressor         | 0.6660            | False                    | 0.6551             | False                     |
| Ridge                         | 0.0421            | False                    | 0.3320             | False                     |
| RidgeCV                       | 0.5513            | False                    | 0.4815             | False                     |
| SGDRegressor                  | 0.7422            | False                    | 0.4762             | False                     |
| SVR                           | 0.1223            | False                    | 0.1778             | False                     |
| TransformedTargetRegressor    | 0.1857            | False                    | 0.2774             | False                     |
| TweedieRegressor              | 0.9460            | False                    | 0.9126             | False                     |
| XGBRegressor                  | 0.8657            | False                    | 0.6529             | False                     |

*P-values rounded to 4 decimal places to improve readability.*

</details>

## Results
Here we compare the results of this implementation with the original. The benchmarks were done on an i5-10400F with 2x8 GB RAM @ 2666 MHz.

<details>
    <summary>Spoiler</summary>
Firstly, we can compare the average time taken for the computation of features for an entire dataset.
<br><br>

| Implementation | Average time taken (seconds) |
|----------------|------------------------------|
| MATLAB         | 140.1177001453079            |
| pointpca2-rs   | 7.261543959379196            |

We can also calculate the absolute differences between corresponding features and then determine the maximum absolute difference. Additionally, we can compute the standard deviation of these absolute differences and find the highest standard deviation among them.

| Maximum absolute difference | Maximum standard deviation |
|-----------------------------|----------------------------|
| 0.11058533454473118         | 0.027662647255776825       |

Finally, we compare the correlation indices, splitting the dataset and fitting the features similarly to the previous section.

<img src="https://i.imgur.com/td9a2wp.png">
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

