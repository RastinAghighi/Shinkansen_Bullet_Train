"""
THE KITCHEN SINK: Advanced Feature Engineering & Mega-Ensemble (FIXED)
Target: Squeeze every last drop of signal (0.96+)
"""

import pandas as pd
import numpy as np
import zipfile
import os
import sys
import warnings

# --- 1. ENABLE EXPERIMENTAL FEATURES FIRST (The Unlock Key) ---
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer

# --- 2. STANDARD IMPORTS ---
try:
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, QuantileTransformer
    from sklearn.linear_model import BayesianRidge
except ImportError as e:
    print(f"[ERROR] Missing Library: {e}")
    print("Please run: pip install lightgbm xgboost catboost scikit-learn")
    sys.exit(1)

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SUBMISSION_FILE = 'Submission_KitchenSink_Mega.csv'
# MATCH YOUR FOLDER PATH
ZIP_PATH = os.path.join('data', 'Olympus', 'archive.zip')

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
        print(f"[ERROR] Could not load data: {e}")
        sys.exit(1)
    
    train = pd.merge(data['train_travel'], data['train_survey'], on='ID')
    test = pd.merge(data['test_travel'], data['test_survey'], on='ID')
    
    target = train['Overall_Experience']
    test_ids = test['ID']
    
    train.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test.drop(['ID'], axis=1, inplace=True)
    
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    return df, target, test_ids, len(train)

def feature_engineering(df):
    print("[INFO] Phase 1: Basic Feature Engineering...")
    # 1. Delay Features
    # Note: Using underscore names based on your new data format
    df['Total_Delay'] = df['Departure_Delay_in_Mins'].fillna(0) + df['Arrival_Delay_in_Mins'].fillna(0)
    df['Delay_Ratio'] = df['Arrival_Delay_in_Mins'] / (df['Departure_Delay_in_Mins'] + 1)
    df['Delay_Interaction'] = df['Departure_Delay_in_Mins'] * df['Arrival_Delay_in_Mins']
    
    # 2. Binning Age
    df['Age_Group'] = pd.cut(df['Age'].fillna(df['Age'].mean()), bins=[0, 18, 30, 50, 100], labels=[0, 1, 2, 3])
    
    # 3. Encoding
    rating_map = {'Excellent': 5, 'Good': 4, 'Acceptable': 3, 'Needs Improvement': 2, 'Poor': 1, 'Satisfied': 1, 'Not Satisfied': 0}
    for col in df.columns:
        if df[col].dtype == 'object':
             if df[col].iloc[0] in rating_map or 'Good' in str(df[col].unique()):
                 df[col] = df[col].map(rating_map)
             else:
                 df[col] = pd.factorize(df[col])[0]
    
    return df

def advanced_processing(df):
    print("[INFO] Phase 2: Advanced Processing (Imputation)...")
    # Impute
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=42)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    print("[INFO] Phase 3: Generating Cluster Features (K-Means)...")
    # Standardize first for Clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_imputed)
    
    kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
    df_imputed['Cluster_ID'] = kmeans.fit_predict(scaled_data)
    
    # Add distance to centroids as features
    distances = kmeans.transform(scaled_data)
    for i in range(7):
        df_imputed[f'Dist_Centroid_{i}'] = distances[:, i]
        
    print("[INFO] Phase 4: Generating PCA Features...")
    pca = PCA(n_components=5, random_state=42)
    pca_features = pca.fit_transform(scaled_data)
    for i in range(5):
        df_imputed[f'PCA_{i}'] = pca_features[:, i]
        
    return df_imputed

def main():
    df, y, ids, train_len = load_data()
    df = feature_engineering(df)
    df = advanced_processing(df)
    
    X = df.iloc[:train_len]
    X_test = df.iloc[train_len:]
    
    print(f"\n[INFO] Final Feature Count: {X.shape[1]} (Started with ~24)")
    
    # --- THE MEGA ENSEMBLE ---
    print("\n[INFO] Initializing The Council of 7...")
    
    # 1. CatBoost (The Champion)
    cat = CatBoostClassifier(iterations=2000, depth=8, learning_rate=0.05, verbose=0, thread_count=-1)
    
    # 2. XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=2000, max_depth=6, learning_rate=0.02, n_jobs=-1, verbosity=0)
    
    # 3. LightGBM
    lgbm = lgb.LGBMClassifier(n_estimators=2000, num_leaves=32, learning_rate=0.03, n_jobs=-1, verbose=-1)
    
    # 4. Random Forest (Diversity)
    rf = RandomForestClassifier(n_estimators=500, max_depth=10, n_jobs=-1, random_state=42)
    
    # 5. Extra Trees (More Diversity)
    et = ExtraTreesClassifier(n_estimators=500, max_depth=10, n_jobs=-1, random_state=42)
    
    # 6. Gradient Boosting (sklearn version)
    gb = GradientBoostingClassifier(n_estimators=500, max_depth=5, random_state=42)
    
    # 7. Logistic Regression (Linear Baseline)
    lr = LogisticRegression(max_iter=1000)

    # --- STACKING ---
    print("[INFO] Training the Meta-Stacking Model...")
    
    estimators = [
        ('cat', cat), ('xgb', xgb_model), ('lgbm', lgbm),
        ('rf', rf), ('et', et), ('gb', gb), ('lr', lr)
    ]
    
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5, # 5-Fold Cross Validation
        n_jobs=-1,
        passthrough=False,
        verbose=1
    )
    
    stack.fit(X, y)
    
    print("\n[INFO] Generating Final Predictions...")
    preds = stack.predict(X_test)
    
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    
    print(f"\n[VICTORY] Saved {SUBMISSION_FILE}")
    print("This file contains the combined intelligence of 7 different algorithms.")

if __name__ == "__main__":
    main()