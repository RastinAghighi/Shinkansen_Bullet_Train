"""
VERSION 1: The Baseline (Optuna CatBoost) - FIXED COLUMN NAMES
Target: Safe, Fast, Reliable Score.
"""

import pandas as pd
import numpy as np
import zipfile
import os
import sys
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from catboost import CatBoostClassifier

# --- CONFIGURATION ---
SUBMISSION_FILE = 'Submission_Version1_Baseline.csv'
ZIP_PATH = os.path.join('data', 'Olympus', 'archive.zip') 

FILES = {
    'train_travel': 'Traveldata_train_(1).csv',
    'train_survey': 'Surveydata_train_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

# WINNING PARAMS (Trial 26)
PARAMS = {
    'iterations': 2873, 'depth': 8, 'learning_rate': 0.078, 
    'l2_leaf_reg': 1.96, 'border_count': 152,
    'loss_function': 'Logloss', 'verbose': 100, 'thread_count': -1
}

def load_data():
    print(f"[INFO] Loading Data from {ZIP_PATH}...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            data = {k: pd.read_csv(z.open(v)) for k, v in FILES.items()}
    except FileNotFoundError:
        print(f"[ERROR] Could not find {ZIP_PATH}")
        sys.exit(1)
        
    train = pd.merge(data['train_travel'], data['train_survey'], on='ID')
    test = pd.merge(data['test_travel'], data['test_survey'], on='ID')
    
    target = train['Overall_Experience']
    test_ids = test['ID']
    
    train.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test.drop(['ID'], axis=1, inplace=True)
    
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # --- FIX IS HERE: Added underscores to column names ---
    df['Total_Delay'] = df['Departure_Delay_in_Mins'].fillna(0) + df['Arrival_Delay_in_Mins'].fillna(0)
    df['Delay_Ratio'] = df['Arrival_Delay_in_Mins'] / (df['Departure_Delay_in_Mins'] + 1)
    
    rating_map = {'Excellent': 5, 'Good': 4, 'Acceptable': 3, 'Needs Improvement': 2, 'Poor': 1, 'Satisfied': 1, 'Not Satisfied': 0}
    for col in df.columns:
        if df[col].dtype == 'object':
             if df[col].iloc[0] in rating_map or 'Good' in str(df[col].unique()):
                 df[col] = df[col].map(rating_map)
             else:
                 df[col] = pd.factorize(df[col])[0]

    print("[INFO] Running MICE Imputation...")
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=42)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    X = df_imputed.iloc[:len(train)]
    X_test = df_imputed.iloc[len(train):]
    y = target
    
    return X, y, X_test, test_ids

def main():
    X, y, X_test, ids = load_data()
    
    print(f"\n[INFO] Training Baseline Model (Trial 26 Configuration)...")
    model = CatBoostClassifier(**PARAMS)
    model.fit(X, y)
    
    print("[INFO] Generating Predictions...")
    preds = model.predict(X_test)
    
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    
    print(f"\n[VICTORY] Generated {SUBMISSION_FILE}")
    print(">> UPLOAD THIS FILE FIRST <<")

if __name__ == "__main__":
    main()