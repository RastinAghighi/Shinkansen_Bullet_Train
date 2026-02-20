"""
THE REFINERY: Feature Engineering & Data Cleaning
Target: 0.960+ (Better Data > Better Models)
"""
import pandas as pd
import numpy as np
import os
import sys
import zipfile
from catboost import CatBoostClassifier

# --- 1. SETUP ---
SUBMISSION_FILE = 'Submission_Feature_Eng.csv'

# AUTOMATIC DATA FINDER (Kaggle/Local)
print("[INFO] Hunting for data...")
DATA_DIR = ""
for root, dirs, files in os.walk('.'):
    if "Traveldata_train_(1).csv" in files:
        DATA_DIR = root
        break
for root, dirs, files in os.walk('/kaggle/input'):
    if "Traveldata_train_(1).csv" in files:
        DATA_DIR = root
        break

if not DATA_DIR:
    print("[ERROR] Could not find data!")
    sys.exit(1)

# --- 2. THE CHAMPION PARAMETERS (Trial 0) ---
PARAMS = {
    'iterations': 1200, 
    'depth': 10, 
    'learning_rate': 0.07167, 
    'l2_leaf_reg': 6.357, 
    'border_count': 35, 
    'random_strength': 8.32, 
    'bagging_temperature': 0.69,
    'loss_function': 'Logloss', 
    'verbose': 100,
    'thread_count': -1,
    # 'task_type': 'GPU', # Uncomment for Kaggle
    # 'devices': '0'
}

# --- 3. FEATURE ENGINEERING LOGIC ---
def engineer_features(df):
    print("   -> Generating 'Smart' Features...")
    
    # A. The "Vibe" Features (Aggregating Survey Ratings)
    # Map strings to numbers JUST for calculation (CatBoost still sees strings later)
    rating_map = {'Excellent': 5, 'Good': 4, 'Acceptable': 3, 'Needs Improvement': 2, 'Poor': 1, 'Missing_Data': 0}
    
    survey_cols = [c for c in df.columns if df[c].dtype == 'object' and 'Excellent' in str(df[c].unique())]
    
    # Create temporary numeric block
    temp_df = df[survey_cols].replace(rating_map)
    for col in temp_df.columns:
        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0)
    
    # 1. Total Satisfaction Score
    df['FE_Total_Rating'] = temp_df.sum(axis=1)
    
    # 2. Average Rating
    df['FE_Mean_Rating'] = temp_df.mean(axis=1)
    
    # 3. Polarity (Standard Deviation)
    # Does the user hate everything (low std) or love/hate specific things (high std)?
    df['FE_Rating_Std'] = temp_df.std(axis=1)
    
    # 4. The "Missing" Count (How many questions did they skip?)
    df['FE_Missing_Count'] = (df[survey_cols] == 'Missing_Data').sum(axis=1)
    
    # B. The "Pain" Features (Delays)
    df['Total_Delay'] = df['Departure_Delay_in_Mins'].fillna(0) + df['Arrival_Delay_in_Mins'].fillna(0)
    
    # 5. Delay Severity (Delay / Distance)
    # 30 min delay on 100 mile flight (BAD) vs 30 min delay on 5000 mile flight (OK)
    df['FE_Delay_Per_Mile'] = df['Total_Delay'] / (df['Travel_Distance'] + 1)
    
    return df

def clean_training_data(X, y):
    print("\n[PURGE] Scanning for Contradictory Rows...")
    original_len = len(X)
    
    # Re-calculate ratings just for filtering
    rating_map = {'Excellent': 5, 'Good': 4, 'Acceptable': 3, 'Needs Improvement': 2, 'Poor': 1, 'Missing_Data': 0}
    survey_cols = [c for c in X.columns if X[c].dtype == 'object' and 'Excellent' in str(X[c].unique())]
    temp_df = X[survey_cols].replace(rating_map)
    for col in temp_df.columns:
        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0)
        
    avg_rating = temp_df.mean(axis=1)
    
    # LOGIC 1: The "Hater" Paradox
    # User says "Overall Satisfied" (1) but Average Rating < 2.0 (Mostly Poor)
    mask_hater = (y == 1) & (avg_rating < 1.5)
    
    # LOGIC 2: The "Lover" Paradox
    # User says "Not Satisfied" (0) but Average Rating > 4.5 (Mostly Excellent)
    mask_lover = (y == 0) & (avg_rating > 4.5)
    
    # Combine masks
    bad_rows = mask_hater | mask_lover
    
    X_clean = X[~bad_rows]
    y_clean = y[~bad_rows]
    
    print(f"   -> Dropped {sum(bad_rows)} rows ({sum(bad_rows)/original_len:.1%}) that didn't make sense.")
    return X_clean, y_clean

def load_data():
    FILES = {
        'train_travel': 'Traveldata_train_(1).csv',
        'train_survey': 'Surveydata_train_(1).csv',
        'test_travel': 'Traveldata_test_(1).csv',
        'test_survey': 'Surveydata_test_(1).csv'
    }
    
    data = {k: pd.read_csv(os.path.join(DATA_DIR, v)) for k, v in FILES.items()}

    train = pd.merge(data['train_travel'], data['train_survey'], on='ID')
    test = pd.merge(data['test_travel'], data['test_survey'], on='ID')
    
    target = train['Overall_Experience']
    test_ids = test['ID']
    train.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test.drop(['ID'], axis=1, inplace=True)
    
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # --- APPLY FEATURE ENGINEERING ---
    df = engineer_features(df)
    
    # Prepare Categoricals for Native Mode
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Missing_Data").astype(str)
            cat_cols.append(col)
        else:
            df[col] = df[col].fillna(-999)

    return df.iloc[:len(train)], target, df.iloc[len(train):], test_ids, cat_cols

def main():
    X, y, X_test, ids, cat_cols = load_data()
    
    # --- APPLY DATA CLEANING (SAMPLING) ---
    X_clean, y_clean = clean_training_data(X, y)
    
    print(f"\n[INFO] Training Champion Model on {len(X_clean)} rows...")
    model = CatBoostClassifier(**PARAMS, cat_features=cat_cols)
    model.fit(X_clean, y_clean)
    
    print("[INFO] Generating Predictions...")
    preds = model.predict(X_test)
    
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    
    print(f"\n[VICTORY] Generated {SUBMISSION_FILE}")

if __name__ == "__main__":
    main()