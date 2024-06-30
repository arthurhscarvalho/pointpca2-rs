# PointPCA2 - Rust
#### An implementation of PointPCA2 in Rust, with seamless Python integration

This project features a Rust implementation of PointPCA2, designed for faster feature computation and improved RAM management. Additionally, a Python package is provided, enabling the use of Python's comprehensive data science tools. A statistical t-test will be conducted to compare results from the original and Rust implementations, ensuring the validity and reliability of the adaptation.

## Roadmap
- [x] Implementation: a fully working version of PointPCA2, written entirely in Rust
- [x] Python package: a Python package that comunicates with the Rust implementation
- [ ] Statistical comparison: conduct a t-test to statistically compare the features generated from the orignal and Rust implementations
- [ ] Distribution: ship the Python package to PyPI

## Repository structure
Please note that the *main* branch encompasses the Rust implementation, and the *pypi* branch encompasses the Python package.

## Setup

### Build using Cargo
For Rust usage, install the Rust programming language in your system, and simply clone this repository and run ```cargo run```.

### Build using Maturin
To use the project as a Python package, firstly setup a virtual environment of your choice and activate it, then clone the "pypi" branch from this project, and install the Python requirements in the *requirements.txt* file. Install the Rust programming language in your system. Run ```maturin build -r``` and the package will be installed in your environment.

### From PyPI
Not yet available.

## Usage
If you only intend to use the Rust part of this project, the *main.rs* file contains an example of the usage. Please keep in mind that the function for reading point clouds is **very** experimental.

If the intended usage is towards the Python package, the file *examples/example_compute_predictors.py* contains an example of the correct usage of the Python module. Note that this file can be found on the PyPI branch.

## Validity
A statistical t-test is on progress to ensure the validity and reliability of this implementation.

## Results
This section is still under construction.

## Contributing
Feel free to open issues with contributions to this project. Any kind of contributions are greatly appreciated.

## References
- [pointpca2](https://github.com/cwi-dis/pointpca2/) - 2023 Grand Challenge on Objective Quality Metrics for Volumetric Contents

## License
GNU GENERAL PUBLIC LICENSE<br>
Version 2, June 1991

---

> GitHub [@akaTsunemori](https://github.com/akaTsunemori)

