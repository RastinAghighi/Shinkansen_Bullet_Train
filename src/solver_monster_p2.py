import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import warnings

# Suppress the warnings so they don't scare us
warnings.filterwarnings('ignore')

SUBMISSION_FILE = 'Submission_Monster_Ensemble.csv'

def load():
    print("[MONSTER] Loading Processed Data...")
    d = np.load('monster_data.npz', allow_pickle=True)
    return d['X'], d['y'], d['X_test'], d['ids']

def train():
    X, y, X_test, ids = load()
    
    print("[MONSTER] Initializing The Council of 7...")
    
    # 1. CatBoost
    m1 = CatBoostClassifier(iterations=2500, depth=8, learning_rate=0.05, verbose=0, thread_count=-1)
    
    # 2. XGBoost
    m2 = xgb.XGBClassifier(n_estimators=2500, max_depth=7, learning_rate=0.015, n_jobs=-1, verbosity=0)
    
    # 3. LightGBM
    m3 = lgb.LGBMClassifier(n_estimators=2500, num_leaves=50, learning_rate=0.02, n_jobs=-1, verbose=-1)
    
    # 4. Random Forest
    m4 = RandomForestClassifier(n_estimators=600, max_depth=12, n_jobs=-1, random_state=42)
    
    # 5. Extra Trees
    m5 = ExtraTreesClassifier(n_estimators=600, max_depth=12, n_jobs=-1, random_state=42)
    
    # 6. Gradient Boosting
    m6 = GradientBoostingClassifier(n_estimators=500, max_depth=5, random_state=42)
    
    # 7. Logistic Regression (FIXED: Increased iterations to 10,000)
    m7 = LogisticRegression(max_iter=10000, C=0.5, solver='liblinear')

    estimators = [
        ('cat', m1), ('xgb', m2), ('lgbm', m3),
        ('rf', m4), ('et', m5), ('gb', m6), ('lr', m7)
    ]
    
    print("[MONSTER] Training Stacking Ensemble (This will take 15-20 mins)...")
    
    # FIXED: The Final Judge also gets 10,000 iterations
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=10000),
        cv=5,
        n_jobs=-1,
        passthrough=True
    )
    
    stack.fit(X, y)
    
    print("[MONSTER] Predicting...")
    preds = stack.predict(X_test)
    
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    print(f"\n[VICTORY] Saved {SUBMISSION_FILE}")
    print(">> UPLOAD THIS FILE TO WIN <<")

if __name__ == "__main__":
    train()