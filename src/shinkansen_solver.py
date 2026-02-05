"""
Shinkansen Passenger Satisfaction Solver

This module implements a machine learning pipeline to predict passenger satisfaction
based on travel data and survey responses. It utilizes an XGBoost classifier with
KNN imputation for missing values.

The pipeline operates in two modes:
1. Validation Mode: Holds out 20% of data to estimate model performance.
2. Production Mode: Retrains on the full dataset to generate final predictions.

Repository: Shinkansen_Solver
"""

import os
import sys
import zipfile
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Configuration ---
SUBMISSION_FILE = 'Submission_XGBoost.csv'
DATA_DIR = 'data'
ZIP_FILENAME = 'archive.zip'
ZIP_PATH = os.path.join(DATA_DIR, ZIP_FILENAME)

# File mappings within the archive
FILES = {
    'train_travel': 'Traveldata_train_(1)_(2).csv',
    'train_survey': 'Surveydata_train_(1)_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

# Model Hyperparameters (Optimized for high-dimensional survey data)
MODEL_PARAMS = {
    'n_estimators': 2500,
    'learning_rate': 0.01,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'n_jobs': -1,
    'random_state': 42
}

def load_data_from_zip(zip_path: str, file_map: Dict[str, str]) -> Tuple[pd.DataFrame, ...]:
    """
    Extracts and loads CSV datasets directly from the specified ZIP archive.

    Args:
        zip_path: Path to the .zip archive containing data files.
        file_map: Dictionary mapping internal logical names to filenames in the zip.

    Returns:
        Tuple of four DataFrames: (train_travel, train_survey, test_travel, test_survey).
    """
    if not os.path.exists(zip_path):
        print(f"[ERROR] Archive not found at: {zip_path}")
        print(f"[HINT] Ensure '{ZIP_FILENAME}' is placed inside the '{DATA_DIR}' directory.")
        sys.exit(1)

    print(f"[INFO] Loading data from {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            data = {key: pd.read_csv(z.open(filename)) for key, filename in file_map.items()}
        return (data['train_travel'], data['train_survey'], 
                data['test_travel'], data['test_survey'])
    except KeyError as e:
        sys.exit(f"[ERROR] Missing file in archive: {e}")
    except Exception as e:
        sys.exit(f"[ERROR] Failed to load data: {e}")

def preprocess_pipeline(train_travel: pd.DataFrame, train_survey: pd.DataFrame, 
                       test_travel: pd.DataFrame, test_survey: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Executes the data processing pipeline: merging, feature engineering, encoding, and imputation.
    """
    print("[INFO] Starting preprocessing pipeline...")

    # 1. Merge Datasets
    train_df = pd.merge(train_travel, train_survey, on='ID')
    test_df = pd.merge(test_travel, test_survey, on='ID')

    # 2. Separate Identifiers and Target
    target = train_df['Overall_Experience']
    test_ids = test_df['ID']
    
    # Drop non-feature columns
    train_df.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test_df.drop(['ID'], axis=1, inplace=True)

    # Combine for consistent encoding/imputation
    combined_df = pd.concat([train_df, test_df], axis=0)

    # 3. Feature Engineering
    # Calculate Total Delay impact
    combined_df['Total_Delay'] = (
        combined_df['DepartureDelay_in_Mins'].fillna(0) + 
        combined_df['ArrivalDelay_in_Mins'].fillna(0)
    )
    # Calculate Delay Ratio (handling division by zero)
    combined_df['Delay_Ratio'] = (
        combined_df['ArrivalDelay_in_Mins'].fillna(0) / 
        (combined_df['DepartureDelay_in_Mins'].fillna(0) + 1.0)
    )

    # 4. Encoding Categorical Variables
    # Map ordinal survey responses to numerical scale
    rating_map = {
        'Excellent': 5, 'Good': 4, 'Acceptable': 3, 
        'Needs Improvement': 2, 'Poor': 1,
        'Satisfied': 1, 'Not Satisfied': 0, 
        np.nan: np.nan
    }

    categorical_cols = combined_df.select_dtypes(include='object').columns
    for col in categorical_cols:
        unique_vals = [str(x) for x in combined_df[col].unique()]
        # Heuristic check for rating columns
        if any('Good' in x for x in unique_vals) or any('Excellent' in x for x in unique_vals):
            combined_df[col] = combined_df[col].map(rating_map)
        else:
            # Factorize nominal columns (Gender, CustomerType, etc.)
            combined_df[col] = pd.factorize(combined_df[col])[0]

    # 5. Imputation (KNN)
    # Uses K-Nearest Neighbors to impute missing survey responses based on similarity
    print("[INFO] Imputing missing values using KNN (k=5)...")
    imputer = KNNImputer(n_neighbors=5)
    imputed_array = imputer.fit_transform(combined_df)
    imputed_df = pd.DataFrame(imputed_array, columns=combined_df.columns)

    # Split back into Train and Test
    X = imputed_df.iloc[:len(train_df), :]
    X_test = imputed_df.iloc[len(train_df):, :]

    return X, target, X_test, test_ids

def run_validation(X: pd.DataFrame, y: pd.Series):
    """
    Runs a hold-out validation test (Mock Exam) to estimate model accuracy.
    """
    print("\n--- Validation Phase ---")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    score = accuracy_score(y_val, preds)
    
    print(f"[RESULT] Estimated Validation Accuracy: {score:.5f} ({score*100:.2f}%)")
    print("------------------------\n")
    return score

def generate_submission(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, test_ids: pd.Series):
    """
    Retrains model on full dataset and generates the final submission CSV.
    """
    print("[INFO] Retraining model on full dataset for production...")
    model = XGBClassifier(**MODEL_PARAMS)
    model.fit(X, y)

    print("[INFO] Generating predictions for test set...")
    predictions = model.predict(X_test)
    
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'Overall_Experience': predictions
    })
    
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"[SUCCESS] Submission file saved to: {SUBMISSION_FILE}")

def main():
    # 1. Load
    raw_data = load_data_from_zip(ZIP_PATH, FILES)
    
    # 2. Preprocess
    X, y, X_test, test_ids = preprocess_pipeline(*raw_data)
    
    # 3. Validate (Mock Test)
    run_validation(X, y)
    
    # 4. Execute Submission
    generate_submission(X, y, X_test, test_ids)

if __name__ == "__main__":
    main()