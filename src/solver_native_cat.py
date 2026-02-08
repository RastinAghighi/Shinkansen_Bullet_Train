"""
THE UNLEASHED CHAMPION: Native Categorical Handling
Target: 0.97 (By fixing the Feature Encoding mistake)
"""

import pandas as pd
import numpy as np
import zipfile
import os
import sys
from catboost import CatBoostClassifier

# --- CONFIGURATION ---
SUBMISSION_FILE = 'Submission_Native_Cat.csv'
ZIP_PATH = os.path.join('data', 'Olympus', 'archive.zip')

FILES = {
    'train_travel': 'Traveldata_train_(1).csv',
    'train_survey': 'Surveydata_train_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

# HYPERPARAMETERS (Trial 26 - The Winner)
# Note: We let CatBoost determine how to handle the strings
PARAMS = {
    'iterations': 3500, # Increased slightly
    'depth': 8, 
    'learning_rate': 0.05, # Lower LR for better convergence
    'l2_leaf_reg': 1.96, 
    'border_count': 152,
    'loss_function': 'Logloss', 
    'verbose': 100, 
    'thread_count': -1,
    'task_type': 'CPU' # Change to GPU if you have one set up
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
    
    # --- FE: Delay Logic ---
    df['Total_Delay'] = df['Departure_Delay_in_Mins'].fillna(0) + df['Arrival_Delay_in_Mins'].fillna(0)
    df['Delay_Ratio'] = df['Arrival_Delay_in_Mins'] / (df['Departure_Delay_in_Mins'] + 1)
    
    # --- CRITICAL CHANGE: DO NOT ENCODE STRINGS TO NUMBERS ---
    # We only fill NaNs with "Missing" so CatBoost handles them as a category
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Missing_Data")
            df[col] = df[col].astype(str) # Ensure it is string
            cat_cols.append(col)
        else:
            # Numerical imputation (simple)
            df[col] = df[col].fillna(-999)

    print(f"[INFO] Identified {len(cat_cols)} Categorical Features to be handled natively.")
    print(cat_cols)
    
    X = df.iloc[:len(train)]
    X_test = df.iloc[len(train):]
    y = target
    
    return X, y, X_test, test_ids, cat_cols

def main():
    X, y, X_test, ids, cat_features = load_data()
    
    print(f"\n[INFO] Training Native CatBoost (Unleashed)...")
    # We pass 'cat_features' here. This triggers the special algorithm.
    model = CatBoostClassifier(**PARAMS, cat_features=cat_features)
    
    model.fit(X, y)
    
    print("[INFO] Generating Predictions...")
    preds = model.predict(X_test)
    
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    
    print(f"\n[VICTORY] Generated {SUBMISSION_FILE}")
    print("This model used the RAW STRINGS. If this doesn't hit 0.96, the data is the limit.")

if __name__ == "__main__":
    main()