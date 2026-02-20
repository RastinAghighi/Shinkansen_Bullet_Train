"""
Judge Validator - The Truth Serum
Checks if the Stacking Ensemble beats the Single CatBoost (0.9588).
"""

import pandas as pd
import numpy as np
import zipfile
import os
import sys

# Install libraries
try:
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
except ImportError:
    os.system('pip install lightgbm xgboost catboost')
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# --- CONFIGURATION ---
DATA_DIR = 'data'
ZIP_PATH = os.path.join(DATA_DIR, 'archive.zip')

# --- THE COUNCIL (Same Params as Solver) ---
CAT_PARAMS = {
    'iterations': 2000, 'depth': 8, 'learning_rate': 0.05, 
    'l2_leaf_reg': 3.0, 'border_count': 128, 'verbose': 0, 'thread_count': -1
}
XGB_PARAMS = {
    'n_estimators': 2000, 'max_depth': 6, 'learning_rate': 0.02,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'n_jobs': -1, 'verbosity': 0
}
LGB_PARAMS = {
    'n_estimators': 2000, 'max_depth': 8, 'learning_rate': 0.03,
    'num_leaves': 32, 'n_jobs': -1, 'verbose': -1
}

FILES = {
    'train_travel': 'Traveldata_train_(1)_(2).csv',
    'train_survey': 'Surveydata_train_(1)_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

def load_data():
    print("[INFO] Loading Data...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        data = {k: pd.read_csv(z.open(v)) for k, v in FILES.items()}
    
    # Merge Training Data Only
    train = pd.merge(data['train_travel'], data['train_survey'], on='ID')
    target = train['Overall_Experience']
    train.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    
    # Feature Engineering
    train['Total_Delay'] = train['DepartureDelay_in_Mins'].fillna(0) + train['ArrivalDelay_in_Mins'].fillna(0)
    train['Delay_Ratio'] = train['ArrivalDelay_in_Mins'] / (train['DepartureDelay_in_Mins'] + 1)
    
    # Encode
    rating_map = {'Excellent': 5, 'Good': 4, 'Acceptable': 3, 'Needs Improvement': 2, 'Poor': 1, 'Satisfied': 1, 'Not Satisfied': 0}
    for col in train.columns:
        if train[col].dtype == 'object':
             if train[col].iloc[0] in rating_map or 'Good' in str(train[col].unique()):
                 train[col] = train[col].map(rating_map)
             else:
                 train[col] = pd.factorize(train[col])[0]

    # MICE Imputation
    print("[INFO] Running MICE Imputation...")
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=42)
    X = pd.DataFrame(imputer.fit_transform(train), columns=train.columns)
    
    return X, target

def main():
    X, y = load_data()
    
    # SPLIT THE DATA (Mock Exam)
    print("\n[INFO] Splitting Data (80% Train / 20% Validation)...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define The Ensemble
    estimators = [
        ('cat', CatBoostClassifier(**CAT_PARAMS)),
        ('xgb', xgb.XGBClassifier(**XGB_PARAMS)),
        ('lgb', lgb.LGBMClassifier(**LGB_PARAMS))
    ]
    
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=3, # Faster CV for validation check
        n_jobs=-1,
        passthrough=False
    )
    
    print("\n[INFO] Training The Judge on 80% of data...")
    stack.fit(X_train, y_train)
    
    print("[INFO] Predicting the hidden 20%...")
    preds = stack.predict(X_val)
    
    score = accuracy_score(y_val, preds)
    print(f"\n🏆 [JUDGE SCORE] Stacking Accuracy: {score:.5f}")
    
    if score > 0.9588:
        print(">>> NEW RECORD! The Ensemble beats the Single Model. <<<")
    else:
        print(f">>> It did not beat CatBoost (0.9588). Stick with the Optuna file. <<<")

if __name__ == "__main__":
    main()