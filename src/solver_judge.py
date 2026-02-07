"""
The Judge - Stacking Ensemble (Final Version)
Target: 0.96+ Accuracy via Meta-Learning
"""

import pandas as pd
import numpy as np
import zipfile
import os
import sys

# Install missing libraries
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

# --- CONFIGURATION ---
SUBMISSION_FILE = 'Submission_Judge_0.96.csv'
DATA_DIR = 'data'
ZIP_PATH = os.path.join(DATA_DIR, 'archive.zip')

# --- THE COUNCIL (Base Models) ---
# We use slightly weaker params to prevent overfitting, allowing the Judge to do the work.
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
    
    train = pd.merge(data['train_travel'], data['train_survey'], on='ID')
    test = pd.merge(data['test_travel'], data['test_survey'], on='ID')
    
    target = train['Overall_Experience']
    test_ids = test['ID']
    
    train.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test.drop(['ID'], axis=1, inplace=True)
    
    # Unified Processing
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # Feature Engineering
    df['Total_Delay'] = df['DepartureDelay_in_Mins'].fillna(0) + df['ArrivalDelay_in_Mins'].fillna(0)
    df['Delay_Ratio'] = df['ArrivalDelay_in_Mins'] / (df['DepartureDelay_in_Mins'] + 1)
    
    # Encoding
    rating_map = {'Excellent': 5, 'Good': 4, 'Acceptable': 3, 'Needs Improvement': 2, 'Poor': 1, 'Satisfied': 1, 'Not Satisfied': 0}
    for col in df.columns:
        if df[col].dtype == 'object':
             if df[col].iloc[0] in rating_map or 'Good' in str(df[col].unique()):
                 df[col] = df[col].map(rating_map)
             else:
                 df[col] = pd.factorize(df[col])[0]

    # MICE Imputation
    print("[INFO] Running MICE Imputation...")
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=42)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    X = df_imputed.iloc[:len(train)]
    X_test = df_imputed.iloc[len(train):]
    y = target
    
    return X, y, X_test, test_ids

def main():
    X, y, X_test, ids = load_data()
    
    # Define Base Models
    estimators = [
        ('cat', CatBoostClassifier(**CAT_PARAMS)),
        ('xgb', xgb.XGBClassifier(**XGB_PARAMS)),
        ('lgb', lgb.LGBMClassifier(**LGB_PARAMS))
    ]
    
    # Define The Judge (Logistic Regression)
    # cv=5 means it trains 5 times to learn the patterns perfectly
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1,
        passthrough=False # The Judge only sees the predictions, not raw data
    )
    
    print("\n[INFO] Training The Stacking Ensemble (This takes time)...")
    stack.fit(X, y)
    
    print("[INFO] Generating Predictions...")
    preds = stack.predict(X_test)
    
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    
    print(f"\n[VICTORY] Saved {SUBMISSION_FILE}")
    print("The Judge has spoken.")

if __name__ == "__main__":
    main()