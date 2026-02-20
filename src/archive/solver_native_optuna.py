"""
THE FINAL OPTIMIZATION: Tuning Native CatBoost
Target: 0.96+ (Combining Best Data Format with Best Parameters)
"""

import pandas as pd
import numpy as np
import zipfile
import os
import sys
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
SUBMISSION_FILE = 'Submission_Native_Optuna.csv'
ZIP_PATH = os.path.join('data', 'Olympus', 'archive.zip')
N_TRIALS = 30 # Number of experiments to run

FILES = {
    'train_travel': 'Traveldata_train_(1).csv',
    'train_survey': 'Surveydata_train_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

def load_data():
    print(f"[INFO] Loading Data from {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        data = {k: pd.read_csv(z.open(v)) for k, v in FILES.items()}
        
    train = pd.merge(data['train_travel'], data['train_survey'], on='ID')
    test = pd.merge(data['test_travel'], data['test_survey'], on='ID')
    
    target = train['Overall_Experience']
    test_ids = test['ID']
    train.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test.drop(['ID'], axis=1, inplace=True)
    
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # --- BASIC PREPROCESSING ONLY (Let CatBoost do the work) ---
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

def objective(trial, X, y, cat_features):
    # Hyperparameter Search Space
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 4000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'od_type': 'Iter',
        'od_wait': 50,
        'loss_function': 'Logloss',
        'verbose': 0,
        'thread_count': -1,
        'task_type': 'CPU', # Change to GPU if available
        'cat_features': cat_features # CRITICAL: Use Native Mode
    }
    
    # 5-Fold Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=50)
        
        preds = model.predict(X_val_fold)
        scores.append(accuracy_score(y_val_fold, preds))
        
    return np.mean(scores)

def main():
    X, y, X_test, ids, cat_features = load_data()
    
    print(f"\n[INFO] Starting Optuna Optimization ({N_TRIALS} Trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y, cat_features), n_trials=N_TRIALS)
    
    print("\n[VICTORY] Best Parameters Found:")
    print(study.best_params)
    
    print("\n[INFO] Training Final Model with Best Params...")
    best_params = study.best_params
    best_params['cat_features'] = cat_features
    best_params['verbose'] = 100
    
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(X, y)
    
    preds = final_model.predict(X_test)
    
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    
    print(f"\n[VICTORY] Generated {SUBMISSION_FILE}")

if __name__ == "__main__":
    main()