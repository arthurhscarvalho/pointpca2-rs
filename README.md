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
| AdaBoostRegressor             | 0.9789            | False                    | 0.8918             | False                     |
| BaggingRegressor              | 0.9361            | False                    | 0.6772             | False                     |
| BayesianRidge                 | 0.9154            | False                    | 0.8313             | False                     |
| DecisionTreeRegressor         | 0.8402            | False                    | 0.6771             | False                     |
| DummyRegressor                | 0.7322            | False                    | 0.5234             | False                     |
| ElasticNet                    | 0.9986            | False                    | 0.9926             | False                     |
| ElasticNetCV                  | 0.9497            | False                    | 0.9439             | False                     |
| ExtraTreeRegressor            | 0.3729            | False                    | 0.3608             | False                     |
| ExtraTreesRegressor           | 0.9184            | False                    | 0.8454             | False                     |
| GammaRegressor                | 0.9910            | False                    | 0.9799             | False                     |
| GaussianProcessRegressor      | 0.9996            | False                    | 0.9869             | False                     |
| GradientBoostingRegressor     | 0.8344            | False                    | 0.9127             | False                     |
| HistGradientBoostingRegressor | 0.9486            | False                    | 0.9218             | False                     |
| HuberRegressor                | 0.7103            | False                    | 0.7560             | False                     |
| KNeighborsRegressor           | 0.9227            | False                    | 0.9489             | False                     |
| KernelRidge                   | 0.9201            | False                    | 0.9187             | False                     |
| LGBMRegressor                 | 0.9693            | False                    | 0.8998             | False                     |
| Lars                          | 0.8300            | False                    | 0.8471             | False                     |
| LarsCV                        | 0.9695            | False                    | 0.9354             | False                     |
| Lasso                         | 0.9997            | False                    | 0.9878             | False                     |
| LassoCV                       | 0.9457            | False                    | 0.9933             | False                     |
| LassoLars                     | 0.9997            | False                    | 0.9918             | False                     |
| LassoLarsCV                   | 0.9412            | False                    | 0.9726             | False                     |
| LassoLarsIC                   | 0.7131            | False                    | 0.7435             | False                     |
| LinearRegression              | 0.5662            | False                    | 0.6076             | False                     |
| LinearSVR                     | 0.9821            | False                    | 0.9684             | False                     |
| MLPRegressor                  | 0.9548            | False                    | 0.9132             | False                     |
| NuSVR                         | 0.9838            | False                    | 0.9637             | False                     |
| OrthogonalMatchingPursuit     | 0.9983            | False                    | 0.9737             | False                     |
| OrthogonalMatchingPursuitCV   | 0.9880            | False                    | 0.9064             | False                     |
| PassiveAggressiveRegressor    | 0.5971            | False                    | 0.5166             | False                     |
| PoissonRegressor              | 0.9931            | False                    | 0.9931             | False                     |
| RANSACRegressor               | 0.7096            | False                    | 0.7028             | False                     |
| RandomForestRegressor         | 0.9017            | False                    | 0.8782             | False                     |
| Ridge                         | 0.9201            | False                    | 0.9187             | False                     |
| RidgeCV                       | 0.9126            | False                    | 0.8655             | False                     |
| SGDRegressor                  | 0.9938            | False                    | 0.9957             | False                     |
| SVR                           | 0.9855            | False                    | 0.9887             | False                     |
| TransformedTargetRegressor    | 0.5662            | False                    | 0.6076             | False                     |
| TweedieRegressor              | 0.9996            | False                    | 0.9990             | False                     |
| XGBRegressor                  | 0.9778            | False                    | 0.8465             | False                     |

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

<img src="https://i.imgur.com/tJBBnPr.png">
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

