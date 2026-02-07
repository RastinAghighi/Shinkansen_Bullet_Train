import pandas as pd
import numpy as np
import zipfile
import os
import sys
import warnings
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import BayesianRidge

warnings.filterwarnings('ignore')

ZIP_PATH = os.path.join('data', 'Olympus', 'archive.zip')
FILES = {
    'train_travel': 'Traveldata_train_(1).csv',
    'train_survey': 'Surveydata_train_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

def get_data():
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        data = {k: pd.read_csv(z.open(v)) for k, v in FILES.items()}
    train = pd.merge(data['train_travel'], data['train_survey'], on='ID')
    test = pd.merge(data['test_travel'], data['test_survey'], on='ID')
    target = train['Overall_Experience']
    ids = test['ID']
    train.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test.drop(['ID'], axis=1, inplace=True)
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    return df, target, ids, len(train)

def process():
    print("[MONSTER] Starting Deep Feature Synthesis...")
    df, y, ids, n_train = get_data()

    df['Total_Delay'] = df['Departure_Delay_in_Mins'].fillna(0) + df['Arrival_Delay_in_Mins'].fillna(0)
    df['Delay_Ratio'] = df['Arrival_Delay_in_Mins'] / (df['Departure_Delay_in_Mins'] + 1)
    df['Delay_Interact'] = df['Departure_Delay_in_Mins'] * df['Arrival_Delay_in_Mins']
    
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
    rating_map = {'Excellent': 5, 'Good': 4, 'Acceptable': 3, 'Needs Improvement': 2, 'Poor': 1, 'Satisfied': 1, 'Not Satisfied': 0}
    
    for c in cat_cols:
        if df[c].iloc[0] in rating_map or 'Good' in str(df[c].unique()):
            df[c] = df[c].map(rating_map)
        else:
            df[c] = pd.factorize(df[c])[0]
            
    print("[MONSTER] Running MICE Imputation...")
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=42)
    df[:] = imputer.fit_transform(df)

    print("[MONSTER] Generating Statistical Aggregates...")
    groups = ['Gender', 'Customer_Type', 'Type_Travel', 'Travel_Class']
    for g in groups:
        df[f'Mean_Age_by_{g}'] = df.groupby(g)['Age'].transform('mean')
        df[f'Mean_Delay_by_{g}'] = df.groupby(g)['Total_Delay'].transform('mean')
        df[f'Std_Delay_by_{g}'] = df.groupby(g)['Total_Delay'].transform('std')

    print("[MONSTER] Generating Clustering Features...")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(scaled)
    dists = kmeans.transform(scaled)
    for i in range(8):
        df[f'Dist_K{i}'] = dists[:, i]

    print("[MONSTER] Generating PCA Features...")
    pca = PCA(n_components=6, random_state=42)
    comps = pca.fit_transform(scaled)
    for i in range(6):
        df[f'PCA_{i}'] = comps[:, i]

    print("[MONSTER] Generating Polynomial Interaction Features...")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    important = ['Total_Delay', 'Age', 'Travel_Distance']
    poly_feats = poly.fit_transform(df[important])
    for i in range(poly_feats.shape[1]):
        df[f'Poly_{i}'] = poly_feats[:, i]

    print(f"[MONSTER] Feature Count: {df.shape[1]}")
    
    X = df.iloc[:n_train]
    X_test = df.iloc[n_train:]
    
    np.savez_compressed('monster_data.npz', X=X, y=y, X_test=X_test, ids=ids)
    print("[MONSTER] Part 1 Complete. Data Saved.")

if __name__ == "__main__":
    process()