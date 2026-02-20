"""
THE CHOSEN ONE: Native CatBoost (Trial 0 Params)
Target: 0.9589+ (Beating your 0.9584 record)
"""

import pandas as pd
import numpy as np
import zipfile
import os
import sys
from catboost import CatBoostClassifier

# --- CONFIGURATION ---
SUBMISSION_FILE = 'Submission_Native_Trial0.csv'
ZIP_PATH = os.path.join('data', 'Olympus', 'archive.zip')
DATA_DIR = os.path.join('data', 'Olympus')

FILES = {
    'train_travel': 'Traveldata_train_(1).csv',
    'train_survey': 'Surveydata_train_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

# --- THE LUCKY PARAMETERS (FROM YOUR LOGS) ---
# Trial 0 value: 0.95891
PARAMS = {
    'iterations': 1108, 
    'depth': 10,  # Deep trees = Smarter logic (but slower)
    'learning_rate': 0.07167235671971785, 
    'l2_leaf_reg': 6.357509098920716, 
    'border_count': 35, 
    'random_strength': 8.32166357055587, 
    'bagging_temperature': 0.6897272791795312,
    'loss_function': 'Logloss', 
    'verbose': 100, 
    'thread_count': -1,
    'task_type': 'CPU'
}

def load_data():
    print("[INFO] Loading Data...")
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            data = {k: pd.read_csv(z.open(v)) for k, v in FILES.items()}
    elif os.path.exists(os.path.join(DATA_DIR, FILES['train_travel'])):
        data = {k: pd.read_csv(os.path.join(DATA_DIR, v)) for k, v in FILES.items()}
    else:
        # Fallback
        data = {k: pd.read_csv(v) for k, v in FILES.items()}

    train = pd.merge(data['train_travel'], data['train_survey'], on='ID')
    test = pd.merge(data['test_travel'], data['test_survey'], on='ID')
    
    target = train['Overall_Experience']
    test_ids = test['ID']
    train.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test.drop(['ID'], axis=1, inplace=True)
    
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # --- PREPROCESSING (Native Mode) ---
    df['Total_Delay'] = df['Departure_Delay_in_Mins'].fillna(0) + df['Arrival_Delay_in_Mins'].fillna(0)
    df['Delay_Ratio'] = df['Arrival_Delay_in_Mins'] / (df['Departure_Delay_in_Mins'] + 1)
    
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # IMPORTANT: Fill NaNs with a string so CatBoost sees them
            df[col] = df[col].fillna("Missing_Data").astype(str)
            cat_cols.append(col)
        else:
            df[col] = df[col].fillna(-999)

    X = df.iloc[:len(train)]
    X_test = df.iloc[len(train):]
    y = target
    
    return X, y, X_test, test_ids, cat_cols

def main():
    X, y, X_test, ids, cat_features = load_data()
    
    print(f"\n[INFO] Training Best Model (Trial 0 Params)...")
    # Native Mode Enabled
    model = CatBoostClassifier(**PARAMS, cat_features=cat_features)
    
    model.fit(X, y)
    
    print("[INFO] Generating Predictions...")
    preds = model.predict(X_test)
    
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    
    print(f"\n[VICTORY] Generated {SUBMISSION_FILE}")
    print("This model scored 0.95891 in Cross-Validation.")
    print("Upload this IMMEDIATELY.")

if __name__ == "__main__":
    main()