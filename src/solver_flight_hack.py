"""
THE FLIGHT HACK: Grouping Passengers by Itinerary
Target: 0.97 (By using peer pressure/group sentiment)
"""

import pandas as pd
import numpy as np
import zipfile
import os
import sys
from catboost import CatBoostClassifier

# --- CONFIGURATION ---
SUBMISSION_FILE = 'Submission_Flight_Hack.csv'
ZIP_PATH = os.path.join('data', 'Olympus', 'archive.zip')

FILES = {
    'train_travel': 'Traveldata_train_(1).csv',
    'train_survey': 'Surveydata_train_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

PARAMS = {
    'iterations': 3500, 'depth': 8, 'learning_rate': 0.05, 
    'l2_leaf_reg': 1.96, 'border_count': 152,
    'loss_function': 'Logloss', 'verbose': 100, 
    'thread_count': -1, 'task_type': 'CPU'
}

def load_data():
    print(f"[INFO] Loading Data from {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        data = {k: pd.read_csv(z.open(v)) for k, v in FILES.items()}
        
    train = pd.merge(data['train_travel'], data['train_survey'], on='ID')
    test = pd.merge(data['test_travel'], data['test_survey'], on='ID')
    
    target = train['Overall_Experience']
    test_ids = test['ID']
    train.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test.drop(['ID'], axis=1, inplace=True)
    
    # Concatenate to find all passengers on the same flight
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # --- THE HACK: RECONSTRUCT FLIGHT ID ---
    # People with same distance and delays are on the same plane.
    # We convert delays to string to handle NaNs safely as a category
    df['Dep_Str'] = df['Departure_Delay_in_Mins'].fillna(-999).astype(str)
    df['Arr_Str'] = df['Arrival_Delay_in_Mins'].fillna(-999).astype(str)
    df['Dist_Str'] = df['Travel_Distance'].astype(str)
    
    # Create a unique signature for each "Flight"
    df['Flight_Signature'] = df['Dist_Str'] + "_" + df['Dep_Str'] + "_" + df['Arr_Str']
    
    print(f"[INFO] Found {df['Flight_Signature'].nunique()} unique flights in the dataset.")
    
    # --- CALCULATE FLIGHT SATISFACTION (Target Encoding with Leakage Protection) ---
    # We want to know: "What is the average satisfaction of THIS flight?"
    # IMPORTANT: We must only use Training labels to teach the model.
    
    # 1. Map Train Labels back to the combined dataframe
    df['Temp_Target'] = np.nan
    df.iloc[:len(train), df.columns.get_loc('Temp_Target')] = target
    
    # 2. Calculate Mean Target per Flight Signature
    # We use a global mean for smoothing rare flights
    global_mean = target.mean()
    flight_means = df.groupby('Flight_Signature')['Temp_Target'].mean()
    
    # 3. Map it back
    df['Flight_Satisfaction_Mean'] = df['Flight_Signature'].map(flight_means)
    
    # 4. Handle Missing (Flights that only appear in Test or have no labels)
    df['Flight_Satisfaction_Mean'] = df['Flight_Satisfaction_Mean'].fillna(global_mean)
    
    # 5. Clean up helper columns
    df.drop(['Dep_Str', 'Arr_Str', 'Dist_Str', 'Flight_Signature', 'Temp_Target'], axis=1, inplace=True)
    
    # --- STANDARD PREPROCESSING (Native Cat Mode) ---
    df['Total_Delay'] = df['Departure_Delay_in_Mins'].fillna(0) + df['Arrival_Delay_in_Mins'].fillna(0)
    
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Missing_Data").astype(str)
            cat_cols.append(col)
        else:
            df[col] = df[col].fillna(-999)

    X = df.iloc[:len(train)]
    X_test = df.iloc[len(train):]
    y = target
    
    return X, y, X_test, test_ids, cat_cols

def main():
    X, y, X_test, ids, cat_features = load_data()
    
    print(f"\n[INFO] Training CatBoost with FLIGHT HACK features...")
    model = CatBoostClassifier(**PARAMS, cat_features=cat_features)
    model.fit(X, y)
    
    print("[INFO] Generating Predictions...")
    preds = model.predict(X_test)
    
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    
    print(f"\n[VICTORY] Generated {SUBMISSION_FILE}")
    print("If 0.97 is possible, this is how they did it.")

if __name__ == "__main__":
    main()