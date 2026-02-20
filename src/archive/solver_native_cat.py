"""
CatBoost Classifier training pipeline for Shinkansen passenger satisfaction.
Utilizes native categorical feature handling via Ordered Target Statistics.
"""

import pandas as pd
import numpy as np
import zipfile
import os
import sys
from catboost import CatBoostClassifier

SUBMISSION_FILE = 'Submission_Native_Cat.csv'
ZIP_PATH = os.path.join('data', 'Olympus', 'archive.zip')

FILES = {
    'train_travel': 'Traveldata_train_(1).csv',
    'train_survey': 'Surveydata_train_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

PARAMS = {
    'iterations': 3500,
    'depth': 8,
    'learning_rate': 0.05,
    'l2_leaf_reg': 1.96,
    'border_count': 152,
    'loss_function': 'Logloss',
    'verbose': 100,
    'thread_count': -1,
    'task_type': 'CPU'
}

def load_data():
    """
    Extracts, merges, and preprocesses training and testing datasets.
    Handles missing values by casting NaNs to a distinct categorical string.
    
    Returns:
        tuple: (X_train, y_train, X_test, test_ids, categorical_features_list)
    """
    print(f"[INFO] Loading Data from {ZIP_PATH}...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            data = {k: pd.read_csv(z.open(v)) for k, v in FILES.items()}
    except Exception as e:
        print(f"[ERROR] Failed to load zip file: {e}")
        sys.exit(1)
        
    train = pd.merge(data['train_travel'], data['train_survey'], on='ID')
    test = pd.merge(data['test_travel'], data['test_survey'], on='ID')
    
    target = train['Overall_Experience']
    test_ids = test['ID']
    
    train.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test.drop(['ID'], axis=1, inplace=True)
    
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    df['Total_Delay'] = df['Departure_Delay_in_Mins'].fillna(0) + df['Arrival_Delay_in_Mins'].fillna(0)
    df['Delay_Ratio'] = df['Arrival_Delay_in_Mins'] / (df['Departure_Delay_in_Mins'] + 1)
    
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Preserving non-response as a behavioral signal
            df[col] = df[col].fillna("Missing_Data")
            df[col] = df[col].astype(str)
            cat_cols.append(col)
        else:
            df[col] = df[col].fillna(-999)

    print(f"[INFO] Identified {len(cat_cols)} categorical features.")
    
    X = df.iloc[:len(train)]
    X_test = df.iloc[len(train):]
    y = target
    
    return X, y, X_test, test_ids, cat_cols

def main():
    """
    Initializes and trains the CatBoost model, executing inference on the test set.
    """
    X, y, X_test, ids, cat_features = load_data()
    
    print("[INFO] Initiating CatBoost training...")
    model = CatBoostClassifier(**PARAMS, cat_features=cat_features)
    
    model.fit(X, y)
    
    print("[INFO] Generating predictions...")
    preds = model.predict(X_test)
    
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    
    print(f"[INFO] Submission saved to {SUBMISSION_FILE}")

if __name__ == "__main__":
    main()