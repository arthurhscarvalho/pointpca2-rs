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

| Model                         | p-value (Pearson)    | p_value ≤ 0.01 (Pearson) | p-value (Spearman)  | p_value ≤ 0.01 (Spearman) |
|-------------------------------|----------------------|--------------------------|---------------------|---------------------------|
| AdaBoostRegressor             | 0.8284086364527132   | False                    | 0.43529070408973575 | False                     |
| BaggingRegressor              | 0.7707016273921702   | False                    | 0.4670205329540999  | False                     |
| BayesianRidge                 | 0.11695826429283192  | False                    | 0.2571493993227054  | False                     |
| DecisionTreeRegressor         | 0.2938946168509655   | False                    | 0.5070296913670156  | False                     |
| DummyRegressor                | 0.3005233471757631   | False                    | 0.1368989755449722  | False                     |
| ElasticNet                    | 0.2886465557018853   | False                    | 0.951114552020106   | False                     |
| ElasticNetCV                  | 0.48826237258046634  | False                    | 0.4001145892701406  | False                     |
| ExtraTreeRegressor            | 0.4712771890973475   | False                    | 0.5297601455705473  | False                     |
| ExtraTreesRegressor           | 0.407887155535605    | False                    | 0.12604343979156082 | False                     |
| GammaRegressor                | 0.1311032896830226   | False                    | 0.19493125365146072 | False                     |
| GaussianProcessRegressor      | 0.9629183470454645   | False                    | 0.13383793549467266 | False                     |
| GradientBoostingRegressor     | 0.16757881608368147  | False                    | 0.31762533718742    | False                     |
| HistGradientBoostingRegressor | 0.3126456995762422   | False                    | 0.14607189722817876 | False                     |
| HuberRegressor                | 0.9787003095810891   | False                    | 0.6584477309354118  | False                     |
| KNeighborsRegressor           | 0.19208068054902214  | False                    | 0.19212975056794462 | False                     |
| KernelRidge                   | 0.02041003098987901  | False                    | 0.4711369173508803  | False                     |
| LGBMRegressor                 | 0.4033244274288073   | False                    | 0.7499108275263318  | False                     |
| Lars                          | 0.9952129180474978   | False                    | 0.9284195487224762  | False                     |
| LarsCV                        | 0.38000303587978607  | False                    | 0.3711366650317451  | False                     |
| Lasso                         | 0.7884014892671709   | False                    | 0.21023128321201268 | False                     |
| LassoCV                       | 0.09270779331973336  | False                    | 0.9882617598747514  | False                     |
| LassoLars                     | 0.7883479878219112   | False                    | 0.373900966300059   | False                     |
| LassoLarsCV                   | 0.06549767690508299  | False                    | 0.34451233031527717 | False                     |
| LassoLarsIC                   | 0.4120710049565982   | False                    | 0.6206712328815888  | False                     |
| LinearRegression              | 0.13503399764422538  | False                    | 0.22363011029161506 | False                     |
| LinearSVR                     | 0.6660797117582105   | False                    | 0.9498672276807866  | False                     |
| MLPRegressor                  | 0.19585492574142563  | False                    | 0.20175440252435076 | False                     |
| NuSVR                         | 0.30444927448708864  | False                    | 0.48655361687242926 | False                     |
| OrthogonalMatchingPursuit     | 0.7380736256178406   | False                    | 0.1736952952160131  | False                     |
| OrthogonalMatchingPursuitCV   | 0.831793998426243    | False                    | 0.20254175261791335 | False                     |
| PassiveAggressiveRegressor    | 0.4464081262694889   | False                    | 0.39814598503238247 | False                     |
| PoissonRegressor              | 0.43549375878653634  | False                    | 0.16365045696078057 | False                     |
| RANSACRegressor               | 0.7823209026838688   | False                    | 0.8515729840523149  | False                     |
| RandomForestRegressor         | 0.6716245758080075   | False                    | 0.9506495959258048  | False                     |
| Ridge                         | 0.020410030985603275 | False                    | 0.4711369173508803  | False                     |
| RidgeCV                       | 0.5321969499213692   | False                    | 0.5072994833129556  | False                     |
| SGDRegressor                  | 0.16736215253623704  | False                    | 0.5049938332074873  | False                     |
| SVR                           | 0.09866138877138482  | False                    | 0.07986471706858318 | False                     |
| TransformedTargetRegressor    | 0.13503399764422538  | False                    | 0.22363011029161506 | False                     |
| TweedieRegressor              | 0.9226151351224712   | False                    | 0.9126317375238324  | False                     |
| XGBRegressor                  | 0.7212902441104376   | False                    | 0.6575188213439072  | False                     |

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

