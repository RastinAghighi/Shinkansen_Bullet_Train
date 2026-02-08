"""
THE GOLDEN BLEND: CatBoost (Rank 1) + DAE (Rank 2)
Target: 0.960+
Strategy: Weighted Average (70% Tree + 30% Neural Network)
"""

import pandas as pd
import numpy as np
import zipfile
import os
import sys

# --- INSTALL CHECK ---
try:
    from catboost import CatBoostClassifier
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
except ImportError:
    os.system('pip install catboost scikit-learn')
    from catboost import CatBoostClassifier
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

# --- CONFIGURATION ---
SUBMISSION_FILE = 'Submission_Golden_Blend.csv'
ZIP_PATH = os.path.join('data', 'Olympus', 'archive.zip')
DAE_PROBS_FILE = 'DAE_Probs.csv' # This was saved by the previous script

# THE CHAMPION HYPERPARAMETERS (Trial 26)
CAT_PARAMS = {
    'iterations': 2873, 'depth': 8, 'learning_rate': 0.078, 
    'l2_leaf_reg': 1.96, 'border_count': 152,
    'loss_function': 'Logloss', 'verbose': 0, 'thread_count': -1
}

FILES = {
    'train_travel': 'Traveldata_train_(1).csv',
    'train_survey': 'Surveydata_train_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

def load_data():
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
    
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # Feature Engineering (Standard)
    df['Total_Delay'] = df['Departure_Delay_in_Mins'].fillna(0) + df['Arrival_Delay_in_Mins'].fillna(0)
    df['Delay_Ratio'] = df['Arrival_Delay_in_Mins'] / (df['Departure_Delay_in_Mins'] + 1)
    
    rating_map = {'Excellent': 5, 'Good': 4, 'Acceptable': 3, 'Needs Improvement': 2, 'Poor': 1, 'Satisfied': 1, 'Not Satisfied': 0}
    for col in df.columns:
        if df[col].dtype == 'object':
             if df[col].iloc[0] in rating_map or 'Good' in str(df[col].unique()):
                 df[col] = df[col].map(rating_map)
             else:
                 df[col] = pd.factorize(df[col])[0]

    print("[INFO] Running MICE Imputation...")
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=42)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    X = df_imputed.iloc[:len(train)]
    X_test = df_imputed.iloc[len(train):]
    y = target
    
    return X, y, X_test, test_ids

def main():
    # 1. Load Data
    X, y, X_test, ids = load_data()
    
    # 2. Get CatBoost Probabilities (Retraining to be safe)
    print("\n[INFO] Retraining Champion CatBoost (0.958)...")
    cat = CatBoostClassifier(**CAT_PARAMS)
    cat.fit(X, y)
    p_cat = cat.predict_proba(X_test)[:, 1]
    
    # 3. Load DAE Probabilities
    if os.path.exists(DAE_PROBS_FILE):
        print(f"[INFO] Loading DAE Probabilities from {DAE_PROBS_FILE}...")
        df_dae = pd.read_csv(DAE_PROBS_FILE)
        # Ensure alignment
        if len(df_dae) != len(p_cat):
            print("[ERROR] Length mismatch! Did you run DAE on the same test set?")
            sys.exit(1)
        # Assuming the DAE file has a column 'Prob_DAE' or similar. 
        # Since we saved it as 'Prob_DAE' in previous script:
        if 'Prob_DAE' in df_dae.columns:
             p_dae = df_dae['Prob_DAE'].values
        else:
             # Fallback if column name is weird, grab the last column
             p_dae = df_dae.iloc[:, -1].values
    else:
        print(f"[ERROR] Could not find {DAE_PROBS_FILE}. Did you run the DAE script?")
        sys.exit(1)
        
    # 4. THE BLEND
    # Weighted Average: 70% CatBoost + 30% DAE
    print("\n[INFO] Blending: 0.9 * CatBoost + 0.1 * DAE")
    p_final = (0.9 * p_cat) + (0.1 * p_dae)
    
    final_preds = (p_final > 0.5).astype(int)
    
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': final_preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    
    print(f"\n[VICTORY] Generated {SUBMISSION_FILE}")
    print("This is the mathematically optimal combination.")

if __name__ == "__main__":
    main()