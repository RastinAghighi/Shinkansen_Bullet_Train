"""
THE FINAL BLEND: Confidence Averaging
Target: 0.960+
Strategy: Combine the 0.9579 (Optuna) and 0.9572 (Hydra) using soft probabilities.
"""

import pandas as pd
import numpy as np
import zipfile
import os
import sys

# Install libraries
try:
    from catboost import CatBoostClassifier
except ImportError:
    os.system('pip install catboost')
    from catboost import CatBoostClassifier

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# --- CONFIGURATION ---
SUBMISSION_FILE = 'Submission_Final_Blend_0.96.csv'
ZIP_PATH = os.path.join('data', 'Olympus', 'archive.zip')

FILES = {
    'train_travel': 'Traveldata_train_(1).csv',
    'train_survey': 'Surveydata_train_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

# 1. OPTUNA PARAMS ( The 0.9579 Winner )
OPTUNA_PARAMS = {
    'iterations': 2873, 'depth': 8, 'learning_rate': 0.078, 
    'l2_leaf_reg': 1.96, 'border_count': 152,
    'loss_function': 'Logloss', 'verbose': 0, 'thread_count': -1
}

# 2. HYDRA PARAMS ( The 0.9572 Runner Up )
# We use slightly different params for diversity
HYDRA_PARAMS = {
    'iterations': 3000, 'depth': 7, 'learning_rate': 0.06, 
    'l2_leaf_reg': 3.0, 'border_count': 128,
    'loss_function': 'Logloss', 'verbose': 0, 'thread_count': -1
}

def load_data():
    print(f"[INFO] Loading Data from {ZIP_PATH}...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            data = {k: pd.read_csv(z.open(v)) for k, v in FILES.items()}
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
        
    train = pd.merge(data['train_travel'], data['train_survey'], on='ID')
    test = pd.merge(data['test_travel'], data['test_survey'], on='ID')
    
    target = train['Overall_Experience']
    test_ids = test['ID']
    
    train.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test.drop(['ID'], axis=1, inplace=True)
    
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # Feature Engineering
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
    
    # MODEL 1: OPTUNA
    print(f"\n[INFO] Training Model 1 (Optuna Champion)...")
    m1 = CatBoostClassifier(**OPTUNA_PARAMS)
    m1.fit(X, y)
    probs_1 = m1.predict_proba(X_test)[:, 1] # Get probabilities, not just 0/1
    
    # MODEL 2: HYDRA VARIANT
    print(f"[INFO] Training Model 2 (Hydra Challenger)...")
    m2 = CatBoostClassifier(**HYDRA_PARAMS)
    m2.fit(X, y)
    probs_2 = m2.predict_proba(X_test)[:, 1]
    
    # BLENDING
    print(f"[INFO] Blending Probabilities...")
    # We give slightly more weight (0.6) to the Champion
    final_probs = (0.6 * probs_1) + (0.4 * probs_2)
    
    final_preds = (final_probs > 0.5).astype(int)
    
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': final_preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    
    print(f"\n[VICTORY] Generated {SUBMISSION_FILE}")
    print("This file contains the wisdom of your top 2 models.")

if __name__ == "__main__":
    main()