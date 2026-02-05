import pandas as pd
import numpy as np
import zipfile
import sys
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier

# --- Configuration ---
SUBMISSION_FILE = 'Submission_XGBoost.csv'
ZIP_FILE = 'data/archive.zip'
FILES = {
    'train_travel': 'Traveldata_train_(1)_(2).csv',
    'train_survey': 'Surveydata_train_(1)_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

def load_data(zip_path: str, file_map: dict):
    """Extracts CSVs directly from the source archive."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            data = {k: pd.read_csv(z.open(v)) for k, v in file_map.items()}
        return data['train_travel'], data['train_survey'], data['test_travel'], data['test_survey']
    except (FileNotFoundError, KeyError) as e:
        sys.exit(f"Error loading data: {e}")

def preprocess(train_travel, train_survey, test_travel, test_survey):
    """Merges datasets, engineers features, and imputes missing values."""
    # Merge on ID
    train_df = pd.merge(train_travel, train_survey, on='ID')
    test_df = pd.merge(test_travel, test_survey, on='ID')

    # Separate target and IDs
    target = train_df['Overall_Experience']
    test_ids = test_df['ID']
    
    train_df.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test_df.drop(['ID'], axis=1, inplace=True)

    # Combine for consistent preprocessing
    df = pd.concat([train_df, test_df], axis=0)

    # --- Feature Engineering ---
    # Aggregate delay impact
    df['Total_Delay'] = df['DepartureDelay_in_Mins'].fillna(0) + df['ArrivalDelay_in_Mins'].fillna(0)
    # Relative delay intensity
    df['Delay_Ratio'] = df['ArrivalDelay_in_Mins'].fillna(0) / (df['DepartureDelay_in_Mins'].fillna(0) + 1.0)

    # --- Encoding ---
    rating_map = {
        'Excellent': 5, 'Good': 4, 'Acceptable': 3, 
        'Needs Improvement': 2, 'Poor': 1,
        'Satisfied': 1, 'Not Satisfied': 0, np.nan: np.nan
    }

    for col in df.select_dtypes(include='object').columns:
        if 'Good' in str(df[col].unique()) or 'Excellent' in str(df[col].unique()):
            df[col] = df[col].map(rating_map)
        else:
            df[col] = pd.factorize(df[col])[0]

    # --- Imputation ---
    # KNN used to capture underlying similarity clusters for missing survey data
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return df_imputed.iloc[:len(train_df)], target, df_imputed.iloc[len(train_df):], test_ids

def run_pipeline():
    """Executes full training and inference pipeline."""
    print("Initializing pipeline...")
    
    # Load & Process
    raw_data = load_data(ZIP_FILE, FILES)
    X, y, X_test, test_ids = preprocess(*raw_data)

    # Model Training
    # Hyperparameters optimized for high-dimensional survey data
    print("Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=2500,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X, y)

    # Inference
    print("Generating predictions...")
    preds = model.predict(X_test)
    
    pd.DataFrame({'ID': test_ids, 'Overall_Experience': preds}).to_csv(SUBMISSION_FILE, index=False)
    print(f"Success. Submission saved to {SUBMISSION_FILE}")

if __name__ == "__main__":
    run_pipeline()