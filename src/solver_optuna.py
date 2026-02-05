"""
Shinkansen Solver - The "Optuna Hunter" (CPU Edition)
Strategy: Automated Hyperparameter Tuning on CatBoost.
Target: ~96.0%
"""

import os
import sys
import zipfile
import pandas as pd
import numpy as np
import optuna
from typing import Tuple, Dict

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# --- Configuration ---
SUBMISSION_FILE = 'Submission_Optuna_CatBoost.csv'
DATA_DIR = 'data'
ZIP_PATH = os.path.join(DATA_DIR, 'archive.zip')
N_TRIALS = 50 

FILES = {
    'train_travel': 'Traveldata_train_(1)_(2).csv',
    'train_survey': 'Surveydata_train_(1)_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

def load_data(zip_path: str, file_map: Dict[str, str]) -> Tuple[pd.DataFrame, ...]:
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
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Ordinal Mapping
    rating_map = {
        'Excellent': 5, 'Good': 4, 'Acceptable': 3, 
        'Needs Improvement': 2, 'Poor': 1,
        'Satisfied': 1, 'Not Satisfied': 0
    }
    
    for col in df.columns:
        unique_vals = [str(x) for x in df[col].unique()]
        if any('Good' in x for x in unique_vals):
            df[col] = df[col].map(rating_map)

    # CatBoost Indexing
    cat_indices = []
    for i, col in enumerate(df.columns):
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col], use_na_sentinel=False)[0]
            df[col] = df[col].replace(-1, np.nan)
            cat_indices.append(i)
        elif col in cat_cols or 'Seat' in col or 'Service' in col:
             if col not in cat_indices:
                 cat_indices.append(i)

    # MICE Imputation
    print("[INFO] Running MICE Imputation...")
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    for idx in cat_indices:
        col_name = df.columns[idx]
        df_imputed[col_name] = df_imputed[col_name].astype(int)

    return df_imputed.iloc[:len(train_df)], target, df_imputed.iloc[len(train_df):], test_ids, cat_indices

def objective(trial, X, y, cat_features):
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 3000),
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'loss_function': 'Logloss',
        'eval_metric': 'Accuracy',
        'verbose': False,
        'task_type': 'CPU',      # FORCE CPU
        'thread_count': -1,      # USE ALL CORES
        'random_seed': 42
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_train_fold, y_train_fold, 
            cat_features=cat_features,
            eval_set=(X_val_fold, y_val_fold),
            early_stopping_rounds=50,
            verbose=False
        )
        scores.append(accuracy_score(y_val_fold, model.predict(X_val_fold)))

    return np.mean(scores)

def main():
    print("[INFO] Loading Data...")
    raw = load_data(ZIP_PATH, FILES)
    X, y, X_test, ids, cat_indices = preprocess(*raw)

    print(f"\n[INFO] Starting Optuna Study (CPU Mode)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y, cat_indices), n_trials=N_TRIALS)

    print(f"🏆 BEST TRIAL SCORE: {study.best_value:.5f}")
    
    print("\n[INFO] Training Final Model...")
    best_params = study.best_params
    best_params['task_type'] = 'CPU'
    best_params['thread_count'] = -1
    best_params['cat_features'] = cat_indices
    best_params['verbose'] = 100
    
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(X, y)
    
    pd.DataFrame({'ID': ids, 'Overall_Experience': final_model.predict(X_test)}).to_csv(SUBMISSION_FILE, index=False)
    print(f"\n[SUCCESS] Saved {SUBMISSION_FILE}")

if __name__ == "__main__":
    main()