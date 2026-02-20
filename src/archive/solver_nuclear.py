"""
THE NUCLEAR OPTION: Multi-Seed Ensemble + Threshold Optimization
Target: 0.960+ (Variance Reduction)
"""

import pandas as pd
import numpy as np
import zipfile
import os
import sys
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize_scalar

# --- CONFIGURATION ---
SUBMISSION_FILE = 'Submission_Nuclear_Option.csv'
ZIP_PATH = os.path.join('data', 'Olympus', 'archive.zip')
DATA_DIR = os.path.join('data', 'Olympus')

# NUCLEAR SETTINGS
N_SEEDS = 15      # Train 15 different versions of the universe
N_FOLDS = 5       # Standard CV
SEEDS = [42, 2023, 1990, 777, 1, 123, 888, 1001, 33, 55, 99, 500, 10, 20, 30]

FILES = {
    'train_travel': 'Traveldata_train_(1).csv',
    'train_survey': 'Surveydata_train_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

# THE CHAMPION HYPERPARAMETERS (Native Mode)
PARAMS = {
    'iterations': 3500, 
    'depth': 8, 
    'learning_rate': 0.05, 
    'l2_leaf_reg': 1.96, 
    'border_count': 152,
    'loss_function': 'Logloss', 
    'verbose': 0, 
    'thread_count': -1,
    'task_type': 'CPU' # Switch to GPU if you have one
}

def load_data():
    print("[INFO] Loading Data...")
    # Robust Loader
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            data = {k: pd.read_csv(z.open(v)) for k, v in FILES.items()}
    elif os.path.exists(os.path.join(DATA_DIR, FILES['train_travel'])):
        data = {k: pd.read_csv(os.path.join(DATA_DIR, v)) for k, v in FILES.items()}
    else:
        # Fallback to current dir
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
            df[col] = df[col].fillna("Missing_Data").astype(str)
            cat_cols.append(col)
        else:
            df[col] = df[col].fillna(-999)

    X = df.iloc[:len(train)]
    X_test = df.iloc[len(train):]
    y = target
    
    return X, y, X_test, test_ids, cat_cols

def get_best_threshold(y_true, y_pred_prob):
    # Function to maximize accuracy
    def acc_func(thr):
        return -accuracy_score(y_true, (y_pred_prob > thr).astype(int))
    
    # Minimize negative accuracy
    res = minimize_scalar(acc_func, bounds=(0.4, 0.6), method='bounded')
    return res.x

def main():
    X, y, X_test, ids, cat_cols = load_data()
    
    # Holders for predictions
    oof_preds = np.zeros(len(X))
    test_preds_accum = np.zeros(len(X_test))
    
    print(f"\n[NUCLEAR] Launching {N_SEEDS} Seeds x {N_FOLDS} Folds = {N_SEEDS*N_FOLDS} Models...")
    
    for i, seed in enumerate(SEEDS):
        print(f"\n>> [SEED {seed}] ({i+1}/{N_SEEDS})")
        
        # Stratified K-Fold for this seed
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        
        seed_oof = np.zeros(len(X))
        seed_test = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Update params with current seed
            current_params = PARAMS.copy()
            current_params['random_seed'] = seed
            current_params['cat_features'] = cat_cols
            
            model = CatBoostClassifier(**current_params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
            
            # Predict
            val_probs = model.predict_proba(X_val)[:, 1]
            test_probs = model.predict_proba(X_test)[:, 1]
            
            # Store
            seed_oof[val_idx] = val_probs
            seed_test += test_probs / N_FOLDS
            
            print(f"   Fold {fold+1}: Acc {accuracy_score(y_val, (val_probs>0.5).astype(int)):.5f}")
            
        # Add to global accumulation
        oof_preds += seed_oof / N_SEEDS
        test_preds_accum += seed_test / N_SEEDS

    # --- POST PROCESSING ---
    print("\n[INFO] Optimization Phase...")
    
    # 1. Find Best Threshold on OOF
    best_thr = get_best_threshold(y, oof_preds)
    base_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
    opt_acc = accuracy_score(y, (oof_preds > best_thr).astype(int))
    
    print(f"   Default Threshold (0.500): {base_acc:.6f}")
    print(f"   Optimal Threshold ({best_thr:.4f}): {opt_acc:.6f}")
    print(f"   Gain: +{opt_acc - base_acc:.6f}")
    
    # 2. Apply to Test
    final_class = (test_preds_accum > best_thr).astype(int)
    
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': final_class})
    sub.to_csv(SUBMISSION_FILE, index=False)
    
    print(f"\n[VICTORY] Saved {SUBMISSION_FILE}")
    print("This file contains the averaged wisdom of 75 models.")

if __name__ == "__main__":
    main()