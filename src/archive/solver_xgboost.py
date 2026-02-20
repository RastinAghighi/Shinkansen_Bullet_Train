"""
Shinkansen Solver - Strategy A: XGBoost (The Ferrari)
Score: 0.9512 (Validation)
Architecture: Single robust XGBoost model with aggressive feature engineering.
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
ZIP_PATH = os.path.join(DATA_DIR, 'archive.zip')

FILES = {
    'train_travel': 'Traveldata_train_(1)_(2).csv',
    'train_survey': 'Surveydata_train_(1)_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

def load_data(zip_path: str, file_map: Dict[str, str]) -> Tuple[pd.DataFrame, ...]:
    """Extracts datasets from zip archive."""
    if not os.path.exists(zip_path):
        sys.exit(f"[ERROR] Archive not found: {zip_path}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            data = {k: pd.read_csv(z.open(v)) for k, v in file_map.items()}
        return (data['train_travel'], data['train_survey'], 
                data['test_travel'], data['test_survey'])
    except Exception as e:
        sys.exit(f"[ERROR] Load failed: {e}")

def preprocess(train_travel, train_survey, test_travel, test_survey):
    """Pipeline: Merge, Feature Engineering, Ordinal Encoding, KNN."""
    # Merge
    train_df = pd.merge(train_travel, train_survey, on='ID')
    test_df = pd.merge(test_travel, test_survey, on='ID')

    target = train_df['Overall_Experience']
    test_ids = test_df['ID']
    
    train_df.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test_df.drop(['ID'], axis=1, inplace=True)

    df = pd.concat([train_df, test_df], axis=0)

    # Feature Engineering
    df['Total_Delay'] = df['DepartureDelay_in_Mins'].fillna(0) + df['ArrivalDelay_in_Mins'].fillna(0)
    df['Delay_Ratio'] = df['ArrivalDelay_in_Mins'].fillna(0) / (df['DepartureDelay_in_Mins'].fillna(0) + 1.0)

    # Ordinal Encoding
    rating_map = {
        'Excellent': 5, 'Good': 4, 'Acceptable': 3, 
        'Needs Improvement': 2, 'Poor': 1,
        'Satisfied': 1, 'Not Satisfied': 0, np.nan: np.nan
    }

    for col in df.select_dtypes(include='object').columns:
        unique = [str(x) for x in df[col].unique()]
        if any(x in str(unique) for x in ['Good', 'Excellent']):
            df[col] = df[col].map(rating_map)
        else:
            df[col] = pd.factorize(df[col])[0]

    # Imputation
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return df_imputed.iloc[:len(train_df)], target, df_imputed.iloc[len(train_df):], test_ids

def main():
    # Load
    raw = load_data(ZIP_PATH, FILES)
    X, y, X_test, ids = preprocess(*raw)

    # Validation
    print("--- Validation (XGBoost) ---")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(
        n_estimators=2500, learning_rate=0.01, max_depth=6, 
        subsample=0.7, colsample_bytree=0.7, n_jobs=-1, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    
    score = accuracy_score(y_val, model.predict(X_val))
    print(f"🏆 XGBoost Score: {score:.5f}")

    # Production
    print("--- Retraining for Submission ---")
    model.fit(X, y)
    pd.DataFrame({'ID': ids, 'Overall_Experience': model.predict(X_test)}).to_csv(SUBMISSION_FILE, index=False)
    print(f"File saved: {SUBMISSION_FILE}")

if __name__ == "__main__":
    main()