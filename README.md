# Ariel---Advanced-FGS1-Feature-Engineering-LGBM
ARIEL Exoplanet Transit Depth Prediction Pipeline

This repository contains a comprehensive Python pipeline for the ARIEL Data Challenge 2025, focused on predicting exoplanet transit depths across multiple wavelengths using FGS1 light curve data.

The solution implements extensive time-series feature engineering (including rolling statistics, FFT, autocorrelation, and spatial metrics), trains a multi-output LightGBM model using K-Fold Cross-Validation, and optimizes the predicted uncertainty ($\sigma$) using the competition's Generalized Log-Likelihood (GLL) scoring metric.

Table of Contents

How to Run the Code Locally

1.1. Prerequisites

1.2. Setup the Environment

1.3. Data Preparation and Configuration (MANDATORY)

1.4. Execute the Pipeline

Key Pipeline Components

Expected Outputs and Performance

1. How to Run the Code Locally

Follow these instructions precisely to set up and run the entire data processing, training, and prediction pipeline on your local machine.

1.1. Prerequisites

You must have Python 3.8+ installed on your system.

1.2. Setup the Environment

We strongly recommend using a virtual environment to manage dependencies.

Create and Activate Virtual Environment:

python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows (Command Prompt):
.\venv\Scripts\activate


Install Dependencies:

Install the necessary scientific and machine learning libraries. Note that pyarrow and fastparquet are crucial for reading the large .parquet data files efficiently.

pip install pandas numpy lightgbm scikit-learn tqdm joblib scipy matplotlib pyarrow fastparquet


1.3. Data Preparation and Configuration (MANDATORY)

The script relies on a specific directory structure that mimics the competition environment. You must download the dataset and correctly configure the base path.

Download the Data:
Download the complete dataset (including train/, test/, *.csv files) and place all contents into a single base directory on your machine.

Locate and Configure BASE_PATH:
Open the Python script (your_script_name.py) and find the Config class near the top. You MUST change the BASE_PATH variable to the absolute path of the directory containing your downloaded data.

For example, if your data resides in /Users/john/ariel-data-challenge-2025/:

class Config:
    """
    Central configuration class.
    """
    # !!! CHANGE THIS TO YOUR LOCAL DATA PATH !!!
    BASE_PATH = '/Users/john/ariel-data-challenge-2025/' 

    TRAIN_CSV = f"{BASE_PATH}/train.csv"
    # ... (rest of the paths)



1.4. Execute the Pipeline

Once the dependencies are installed and the BASE_PATH is correctly configured, run the main script from your terminal:

python your_script_name.py


2. Key Pipeline Components

The solution is organized into modular functions handling specific parts of the machine learning process.

Function

Primary Role

Details

extract_enhanced_transit_features

Feature Engineering

Calculates 183 features from the light curves, including rolling statistics, FFT coefficients, autocorrelation, and spatial metrics (e.g., centroid movement).

prepare_data_with_enhanced_fgs1

Data Preprocessing

Manages the loop over all planets, extracts features, joins the results with stellar metadata, and aligns features and targets.

train_validate_multioutput_cv

Model Training

Implements 5-Fold Cross-Validation, training a separate LightGBM regressor for each of the 283 target wavelengths.

fast_gll_score_numpy

Metric Calculation

A NumPy-optimized implementation of the Generalized Log-Likelihood (GLL) scoring metric.

optimize_sigma_per_wavelength

Uncertainty Calibration

Uses scalar optimization (scipy.optimize.minimize_scalar) to find the ideal scaling factor for the predicted uncertainty ($\sigma$) for each wavelength to maximize the GLL score.

create_submission

Inference & Output

Generates final predictions on the test set, applies the optimized $\sigma$ scalers, and aggregates results to create the final submission CSV.

3. Expected Outputs and Performance

Generated Files

Upon successful completion, the script will generate the following files and directories in the execution directory:

File/Folder

Description

submission.csv

The final prediction file containing transit depths (wl_...) and associated uncertainties (sigma_...) for the test set.

trained_models/

A directory containing all the serialized LightGBM models (.pkl files) for each wavelength and CV fold, along with training metadata.

feature_importance.csv

A breakdown of the importance of the 183 extracted features across all trained models.

Performance Note

This pipeline involves heavy I/O operations and complex feature calculations across large time-series data. Be aware of the following:

Runtime: The feature extraction and CV training stages are highly time-consuming. The initial training data preparation took over 90 minutes in the original run.

Hardware: Performance will be significantly better on machines with fast SSD storage and multiple CPU cores due to the data processing demands.
