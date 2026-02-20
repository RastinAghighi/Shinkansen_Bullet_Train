"""
VERSION 3: The Hydra (Pseudo-Labeling) - FIXED
Target: 0.97+ (High Risk / High Reward)
Strategy: Train -> Identify High Confidence -> Teach itself -> Retrain
"""

import pandas as pd
import numpy as np
import zipfile
import os
import sys

# --- FIX: ADD THIS LINE FIRST ---
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer

# Install missing libraries automatically
try:
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
except ImportError:
    print("[INSTALL] Installing LightGBM & XGBoost...")
    os.system('pip install lightgbm xgboost catboost')
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
SUBMISSION_FILE = 'Submission_Hydra_0.97.csv'
# MATCH YOUR FOLDER PATH
ZIP_PATH = os.path.join('data', 'Olympus', 'archive.zip')

PSEUDO_THRESHOLD_HIGH = 0.99  # Only trust if 99% sure it's 1
PSEUDO_THRESHOLD_LOW = 0.01   # Only trust if 99% sure it's 0

# --- HYPERPARAMETERS ---
CAT_PARAMS = {
    'iterations': 2873, 'depth': 8, 'learning_rate': 0.078, 
    'l2_leaf_reg': 1.96, 'border_count': 152,
    'loss_function': 'Logloss', 'verbose': 0, 'thread_count': -1
}
XGB_PARAMS = {
    'n_estimators': 2500, 'max_depth': 7, 'learning_rate': 0.015,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'objective': 'binary:logistic', 'n_jobs': -1, 'verbosity': 0,
    'tree_method': 'hist'
}
LGB_PARAMS = {
    'n_estimators': 3000, 'max_depth': 8, 'learning_rate': 0.02,
    'num_leaves': 64, 'objective': 'binary',
    'metric': 'binary_logloss', 'n_jobs': -1, 'verbose': -1
}

# --- NEW FILE NAMES ---
FILES = {
    'train_travel': 'Traveldata_train_(1).csv',
    'train_survey': 'Surveydata_train_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

def load_and_preprocess():
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
    
    # Concatenate for consistent encoding
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # --- FIXED COLUMN NAMES (Added Underscores) ---
    df['Total_Delay'] = df['Departure_Delay_in_Mins'].fillna(0) + df['Arrival_Delay_in_Mins'].fillna(0)
    df['Delay_Ratio'] = df['Arrival_Delay_in_Mins'] / (df['Departure_Delay_in_Mins'] + 1)
    
    # Encoding
    rating_map = {'Excellent': 5, 'Good': 4, 'Acceptable': 3, 'Needs Improvement': 2, 'Poor': 1, 'Satisfied': 1, 'Not Satisfied': 0}
    for col in df.columns:
        if df[col].dtype == 'object':
             # Check if it looks like a rating
             if df[col].iloc[0] in rating_map or 'Good' in str(df[col].unique()):
                 df[col] = df[col].map(rating_map)
             else:
                 df[col] = pd.factorize(df[col])[0]

    # MICE Imputation
    print("[INFO] Running MICE Imputation...")
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=42)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    X = df_imputed.iloc[:len(train)].copy()
    X_test = df_imputed.iloc[len(train):].copy()
    y = target
    
    return X, y, X_test, test_ids

def train_predict_layer(X, y, X_test):
    # Train 3 Models and get Probabilities
    print("   > Training CatBoost...")
    cat = CatBoostClassifier(**CAT_PARAMS)
    cat.fit(X, y)
    p_cat = cat.predict_proba(X_test)[:, 1]

    print("   > Training XGBoost...")
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X, y)
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]

    print("   > Training LightGBM...")
    lgbm = lgb.LGBMClassifier(**LGB_PARAMS)
    lgbm.fit(X, y)
    p_lgb = lgbm.predict_proba(X_test)[:, 1]
    
    # Average Probabilities (Ensemble)
    avg_prob = (p_cat + p_xgb + p_lgb) / 3
    return avg_prob

def main():
    X, y, X_test, ids = load_and_preprocess()
    
    # --- PHASE 1: GENERATE PSEUDO LABELS ---
    print("\n[PHASE 1] Initial Training & Pseudo-Label Generation")
    probs = train_predict_layer(X, y, X_test)
    
    # Identify confident predictions
    high_conf_indices = np.where((probs > PSEUDO_THRESHOLD_HIGH) | (probs < PSEUDO_THRESHOLD_LOW))[0]
    pseudo_labels = (probs[high_conf_indices] > 0.5).astype(int)
    
    print(f"\n[INFO] Found {len(high_conf_indices)} high-confidence test samples (Pseudo-Labels).")
    print(f"[INFO] Injecting them into Training Data...")
    
    # Augment Training Data
    X_pseudo = X_test.iloc[high_conf_indices]
    y_pseudo = pd.Series(pseudo_labels)
    
    X_augmented = pd.concat([X, X_pseudo], axis=0)
    y_augmented = pd.concat([y, y_pseudo], axis=0)
    
    # --- PHASE 2: RETRAIN ON SUPER DATASET ---
    print("\n[PHASE 2] Retraining Hydra on Augmented Dataset")
    final_probs = train_predict_layer(X_augmented, y_augmented, X_test)
    
    # Final Hard Voting
    final_preds = (final_probs > 0.5).astype(int)
    
    # Save
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': final_preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    print(f"\n[VICTORY] Saved {SUBMISSION_FILE}")
    print("This is the strongest model mathematically possible on this hardware.")

if __name__ == "__main__":
    main()