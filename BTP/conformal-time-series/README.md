### Time Series Prediction Repository

This repository focuses on producing prediction sets for time series data. The methods implemented here guarantee coverage for any sequence, potentially including adversarial ones. The approach taken here is inspired by control systems, introducing a method called Conformal PID Control.

Various methods are implemented in this codebase, including online quantile regression (quantile tracking/P control), adaptive conformal prediction, and more.

This codebase is designed to facilitate easy extension of methods and addition of new datasets. Below are instructions on how to extend the methods and add new datasets.

#### Getting Started

To reproduce the experiments outlined in our paper, follow these steps:

1. Clone this repository.
2. Create a new conda environment:

    ```
    conda create --name pid
    ```

3. Install the required packages:

    ```
    pip install -r requirements.txt
    ```

4. Navigate to the tests directory:

    ```
    cd tests
    ```

5. Run the tests:

    ```
    bash run_tests.sh
    ```

6. Generate plots:

    ```
    bash make_plots.sh
    ```

For the COVID experiment, additional steps are required. You must first run the jupyter notebook located at `conformal-time-series/tests/datasets/covid-ts-proc/statewide/death-forecasting-perstate-lasso-qr.ipynb`. This notebook requires the `deaths.csv` data file, which can be downloaded from the provided Drive link.

#### Adding New Methods

Follow these steps to add new methods:

1. Define the method in the `core/methods.py` file.
2. Edit the config to include your method.
3. Modify `base_test.py` to include your method.

#### Adding New Datasets

To add new datasets, follow these steps:

1. Load and preprocess the dataset. Ensure it is a pandas dataframe with a valid datetime index and at least one column titled `y` representing the target value.
2. Create a config file for the dataset, describing which methods should be run with what parameters.

After completing these steps, you should be able to run the tests and compute the results.

This repository provides a testing infrastructure for online conformal. The infrastructure spawns a parallel process for every dataset, making it efficient to test one method on all datasets with only one command.

For more details, please refer to the codebase and provided documentation.

