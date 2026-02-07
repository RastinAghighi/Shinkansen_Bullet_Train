"""
Hydra Validator - The Mock Exam (FIXED)
Checks the accuracy of the Hydra strategy using a 80/20 split.
"""

import pandas as pd
import numpy as np
import zipfile
import os
import sys

# Install libraries if missing
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
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_DIR = 'data'
ZIP_PATH = os.path.join(DATA_DIR, 'archive.zip')
PSEUDO_THRESHOLD_HIGH = 0.99
PSEUDO_THRESHOLD_LOW = 0.01

# --- HYPERPARAMETERS ---
CAT_PARAMS = {
    'iterations': 2000, 'depth': 8, 'learning_rate': 0.078, 
    'l2_leaf_reg': 1.96, 'border_count': 152, 'loss_function': 'Logloss', 'verbose': 0, 'thread_count': -1
}
XGB_PARAMS = {
    'n_estimators': 2000, 'max_depth': 7, 'learning_rate': 0.015,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'objective': 'binary:logistic', 'n_jobs': -1, 'verbosity': 0
}
LGB_PARAMS = {
    'n_estimators': 2000, 'max_depth': 8, 'learning_rate': 0.02,
    'num_leaves': 64, 'objective': 'binary', 'n_jobs': -1, 'verbose': -1
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
    
    # Clean Target
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

def train_predict_layer(X_train, y_train, X_val):
    print("   > Training CatBoost...")
    cat = CatBoostClassifier(**CAT_PARAMS)
    cat.fit(X_train, y_train)
    p_cat = cat.predict_proba(X_val)[:, 1]

    print("   > Training XGBoost...")
    # FIX: Renamed variable to xgb_model to avoid conflict with import xgboost as xgb
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_train, y_train)
    p_xgb = xgb_model.predict_proba(X_val)[:, 1]

    print("   > Training LightGBM...")
    lgbm = lgb.LGBMClassifier(**LGB_PARAMS)
    lgbm.fit(X_train, y_train)
    p_lgb = lgbm.predict_proba(X_val)[:, 1]
    
    return (p_cat + p_xgb + p_lgb) / 3

def main():
    X, y = load_data()
    
    # SPLIT THE DATA (The "Mock Exam")
    # We hide 20% of the data (X_val) to check our work
    print("\n[INFO] Splitting Data (80% Train / 20% Validation)...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # PHASE 1: Base Model
    print("\n[PHASE 1] Initial Training...")
    probs = train_predict_layer(X_train, y_train, X_val)
    
    # Check Phase 1 Score
    base_preds = (probs > 0.5).astype(int)
    base_score = accuracy_score(y_val, base_preds)
    print(f"\n[RESULT] Phase 1 Accuracy (Before Hydra): {base_score:.5f}")
    
    # GENERATE PSEUDO LABELS (Simulation)
    # In real life, we predict the TEST set. Here, we pretend X_val is the test set.
    high_conf_indices = np.where((probs > PSEUDO_THRESHOLD_HIGH) | (probs < PSEUDO_THRESHOLD_LOW))[0]
    
    if len(high_conf_indices) > 0:
        print(f"\n[INFO] Found {len(high_conf_indices)} high-confidence samples. Injecting back into training...")
        
        # We take the confident predictions from validation and add them to training
        X_pseudo = X_val.iloc[high_conf_indices]
        y_pseudo = pd.Series((probs[high_conf_indices] > 0.5).astype(int)) # Use our OWN prediction
        
        X_aug = pd.concat([X_train, X_pseudo], axis=0)
        y_aug = pd.concat([y_train, y_pseudo], axis=0)
        
        # PHASE 2: Retrain
        print("\n[PHASE 2] Retraining Hydra...")
        final_probs = train_predict_layer(X_aug, y_aug, X_val)
        final_preds = (final_probs > 0.5).astype(int)
        final_score = accuracy_score(y_val, final_preds)
        
        print(f"\n🏆 [FINAL SCORE] Hydra Accuracy: {final_score:.5f}")
        print(f"   Improvement: {final_score - base_score:.5f}")
        
        if final_score > 0.96:
            print(">>> WE BROKE THE 96% BARRIER! <<<")
        else:
            print(">>> Almost there. We need more data or diversity. <<<")
            
    else:
        print("[WARN] Not enough confident predictions to run Hydra.")

if __name__ == "__main__":
    main()