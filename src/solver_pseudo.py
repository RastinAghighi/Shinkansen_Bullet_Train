"""
Shinkansen Solver - The "Nuclear" Option (Pseudo-Labeling + Iterative Imputation)
Strategy: Semi-Supervised Learning (Self-Training)
Target: > 0.96
"""

import os
import sys
import zipfile
import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict

# Advanced Statistical Imputation
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# Models
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

# --- Configuration ---
SUBMISSION_FILE = 'Submission_PseudoLabel.csv'
DATA_DIR = 'data'
ZIP_PATH = os.path.join(DATA_DIR, 'archive.zip')
PSEUDO_CONFIDENCE_THRESHOLD = 0.95  # Only trust predictions with >95% confidence
N_FOLDS = 10  # 10-Fold CV for extreme precision

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
    """
    Advanced Pipeline: 
    - MICE Imputation (BayesianRidge)
    - Interaction Terms
    - Statistical Features
    """
    # Merge
    train_df = pd.merge(train_travel, train_survey, on='ID')
    test_df = pd.merge(test_travel, test_survey, on='ID')

    target = train_df['Overall_Experience']
    test_ids = test_df['ID']
    
    train_df.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test_df.drop(['ID'], axis=1, inplace=True)

    df = pd.concat([train_df, test_df], axis=0)

    # 1. Advanced Feature Engineering
    df['Total_Delay'] = df['DepartureDelay_in_Mins'].fillna(0) + df['ArrivalDelay_in_Mins'].fillna(0)
    df['Delay_Ratio'] = df['ArrivalDelay_in_Mins'].fillna(0) / (df['DepartureDelay_in_Mins'].fillna(0) + 1.0)
    df['Delay_Interaction'] = df['Total_Delay'] * df['Delay_Ratio']

    # 2. Ordinal Encoding
    rating_map = {
        'Excellent': 5, 'Good': 4, 'Acceptable': 3, 
        'Needs Improvement': 2, 'Poor': 1,
        'Satisfied': 1, 'Not Satisfied': 0, np.nan: np.nan
    }
    
    for col in df.select_dtypes(include='object').columns:
        unique = [str(x) for x in df[col].unique()]
        if any(x in str(unique) for x in ['Good', 'Excellent']):
            df[col] = df[col].map(rating_map)
        else:
            df[col] = pd.factorize(df[col])[0]

    # 3. Iterative Imputation (MICE) - Superior to KNN
    # Models each missing value as a function of other features
    print("[INFO] Running Iterative MICE Imputation (Slow but accurate)...")
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # LightGBM Name Sanitation
    df_imputed = df_imputed.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    return df_imputed.iloc[:len(train_df)], target, df_imputed.iloc[len(train_df):], test_ids

def get_model():
    """Returns the optimized Voting Ensemble."""
    xgb = XGBClassifier(
        n_estimators=3000, learning_rate=0.005, max_depth=8, 
        subsample=0.7, colsample_bytree=0.7, n_jobs=-1, random_state=42, verbosity=0
    )
    
    cat = CatBoostClassifier(
        iterations=3000, learning_rate=0.005, depth=8, 
        verbose=0, random_seed=42, allow_writing_files=False
    )
    
    lgbm = LGBMClassifier(
        n_estimators=3000, learning_rate=0.005, max_depth=8,
        subsample=0.7, colsample_bytree=0.7, n_jobs=-1, random_state=42, verbosity=-1
    )

    return VotingClassifier(
        estimators=[('xgb', xgb), ('cat', cat), ('lgbm', lgbm)],
        voting='soft'
    )

def main():
    # 1. Load & Process
    raw = load_data(ZIP_PATH, FILES)
    X, y, X_test, ids = preprocess(*raw)

    model = get_model()

    # 2. Initial Training (Round 1)
    print("\n--- 🟢 Round 1: Initial Training ---")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    initial_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print(f"Base Model Accuracy: {np.mean(initial_scores):.5f} (+/- {np.std(initial_scores):.5f})")
    
    model.fit(X, y)
    
    # 3. Pseudo-Labeling (The Trick)
    print(f"\n--- 🟡 Round 2: Pseudo-Labeling (Confidence > {PSEUDO_CONFIDENCE_THRESHOLD}) ---")
    
    # Predict probabilities on Test Set
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    
    # Find confident predictions
    max_probs = np.max(probs, axis=1)
    confident_indices = np.where(max_probs > PSEUDO_CONFIDENCE_THRESHOLD)[0]
    
    print(f"[INFO] Found {len(confident_indices)} confident test samples out of {len(X_test)}")
    
    # Create Augmented Training Set
    X_pseudo = X_test.iloc[confident_indices]
    y_pseudo = pd.Series(preds[confident_indices], index=X_pseudo.index)
    
    X_augmented = pd.concat([X, X_pseudo], axis=0)
    y_augmented = pd.concat([y, y_pseudo], axis=0)
    
    print(f"[INFO] New Training Size: {len(X_augmented)} (Was {len(X)})")

    # 4. Final Training (Round 2)
    print("\n--- 🔴 Round 3: Final Training on Augmented Data ---")
    final_model = get_model()
    final_model.fit(X_augmented, y_augmented)
    
    # 5. Submission
    final_preds = final_model.predict(X_test)
    pd.DataFrame({'ID': ids, 'Overall_Experience': final_preds}).to_csv(SUBMISSION_FILE, index=False)
    print(f"\n[SUCCESS] Generated {SUBMISSION_FILE} using Pseudo-Labeling.")

if __name__ == "__main__":
    main()