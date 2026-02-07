"""
THE BRAIN: Deep Learning (TensorFlow/Keras)
Target: Diversity. (If this gets >0.94, blending it with CatBoost wins).
"""

import pandas as pd
import numpy as np
import os
import sys
import zipfile

# --- INSTALL CHECK ---
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from sklearn.preprocessing import StandardScaler, QuantileTransformer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
except ImportError:
    print("[INSTALL] Installing TensorFlow & Scikit-Learn...")
    os.system('pip install tensorflow scikit-learn')
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from sklearn.preprocessing import StandardScaler, QuantileTransformer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

# --- CONFIGURATION ---
SUBMISSION_FILE = 'Submission_Neural_Network.csv'
ZIP_PATH = os.path.join('data', 'Olympus', 'archive.zip')

FILES = {
    'train_travel': 'Traveldata_train_(1).csv',
    'train_survey': 'Surveydata_train_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

def load_and_preprocess():
    print(f"[INFO] Loading Data from {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        data = {k: pd.read_csv(z.open(v)) for k, v in FILES.items()}
    
    train = pd.merge(data['train_travel'], data['train_survey'], on='ID')
    test = pd.merge(data['test_travel'], data['test_survey'], on='ID')
    
    target = train['Overall_Experience']
    test_ids = test['ID']
    
    train.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test.drop(['ID'], axis=1, inplace=True)
    
    # Concatenate
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # Feature Engineering
    df['Total_Delay'] = df['Departure_Delay_in_Mins'].fillna(0) + df['Arrival_Delay_in_Mins'].fillna(0)
    df['Delay_Ratio'] = df['Arrival_Delay_in_Mins'] / (df['Departure_Delay_in_Mins'] + 1)
    
    # Encoding (Neural Networks need numbers, not strings)
    rating_map = {'Excellent': 5, 'Good': 4, 'Acceptable': 3, 'Needs Improvement': 2, 'Poor': 1, 'Satisfied': 1, 'Not Satisfied': 0}
    for col in df.columns:
        if df[col].dtype == 'object':
             if df[col].iloc[0] in rating_map or 'Good' in str(df[col].unique()):
                 df[col] = df[col].map(rating_map)
             else:
                 # One-Hot Encoding is often better for NNs, but Factorize is safer for dimensionality here
                 df[col] = pd.factorize(df[col])[0]

    # Imputation
    print("[INFO] Running MICE Imputation...")
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=3, random_state=42)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # SCALING (CRITICAL FOR NEURAL NETWORKS)
    # NNs fail if data isn't between 0 and 1 or -1 and 1
    print("[INFO] Scaling Data (Quantile Transformer)...")
    scaler = QuantileTransformer(output_distribution='normal')
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)
    
    X = df_scaled.iloc[:len(train)]
    X_test = df_scaled.iloc[len(train):]
    y = target
    
    return X, y, X_test, test_ids

def build_model(input_shape):
    # Modern ResNet-style MLP for Tabular Data
    inputs = keras.Input(shape=(input_shape,))
    
    # Block 1
    x = layers.Dense(256, activation="swish")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 2
    x = layers.Dense(128, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Block 3
    x = layers.Dense(64, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    X, y, X_test, ids = load_and_preprocess()
    
    print(f"\n[INFO] Training Neural Network on {X.shape[1]} features...")
    
    model = build_model(X.shape[1])
    
    # Early Stopping to prevent overfitting
    early_stopping = callbacks.EarlyStopping(
        min_delta=0.0005, 
        patience=10, 
        restore_best_weights=True
    )
    
    history = model.fit(
        X, y,
        validation_split=0.2,
        batch_size=512,
        epochs=100,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("\n[INFO] Generating Predictions...")
    probs = model.predict(X_test)
    preds = (probs > 0.5).astype(int).flatten()
    
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    
    val_acc = max(history.history['val_accuracy'])
    print(f"\n[VICTORY] Best Validation Accuracy: {val_acc:.5f}")
    print(f"Saved {SUBMISSION_FILE}")

if __name__ == "__main__":
    main()