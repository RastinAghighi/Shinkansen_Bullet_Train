"""
THE GOD MODE: Deep & Cross Network (DCN) - FIXED
Target: 0.96+ by learning explicit feature interactions.
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
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
except ImportError:
    print("[INSTALL] Installing TensorFlow...")
    os.system('pip install tensorflow scikit-learn')
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

# --- CONFIGURATION ---
SUBMISSION_FILE = 'Submission_DCN_GodMode.csv'
ZIP_PATH = os.path.join('data', 'Olympus', 'archive.zip')

FILES = {
    'train_travel': 'Traveldata_train_(1).csv',
    'train_survey': 'Surveydata_train_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

# --- CUSTOM CROSS LAYER (Fixed for Dimensions) ---
class CrossLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(CrossLayer, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], 1),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(input_shape[1],),
                                    initializer='zeros',
                                    trainable=True)
        super(CrossLayer, self).build(input_shape)

    def call(self, x):
        # FIX: Simplified Dot Product Logic
        # x shape: (Batch, Features)
        # kernel shape: (Features, 1)
        
        # 1. Compute Dot Product (Batch, 1)
        dot_prod = tf.matmul(x, self.kernel)
        
        # 2. Scale Input by Dot Product (Feature Interaction)
        # (Batch, Features) * (Batch, 1) -> (Batch, Features)
        x_interaction = x * dot_prod
        
        # 3. Add Bias and Residual Connection
        return x_interaction + self.bias + x

    def get_config(self):
        config = super(CrossLayer, self).get_config()
        config.update({"output_dim": self.output_dim})
        return config

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
    
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # Feature Engineering
    df['Total_Delay'] = df['Departure_Delay_in_Mins'].fillna(0) + df['Arrival_Delay_in_Mins'].fillna(0)
    df['Delay_Ratio'] = df['Arrival_Delay_in_Mins'] / (df['Departure_Delay_in_Mins'] + 1)
    
    # Encoding
    rating_map = {'Excellent': 5, 'Good': 4, 'Acceptable': 3, 'Needs Improvement': 2, 'Poor': 1, 'Satisfied': 1, 'Not Satisfied': 0}
    for col in df.columns:
        if df[col].dtype == 'object':
             if df[col].iloc[0] in rating_map or 'Good' in str(df[col].unique()):
                 df[col] = df[col].map(rating_map)
             else:
                 df[col] = pd.factorize(df[col])[0]

    # Imputation
    print("[INFO] Running MICE Imputation...")
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=3, random_state=42)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Scaling (Strictly required for DCN)
    print("[INFO] Scaling Data (Quantile)...")
    scaler = QuantileTransformer(output_distribution='normal')
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)
    
    X = df_scaled.iloc[:len(train)]
    X_test = df_scaled.iloc[len(train):]
    y = target
    
    return X, y, X_test, test_ids

def build_dcn_model(input_shape):
    inputs = keras.Input(shape=(input_shape,))
    
    # --- CROSS NETWORK (Explicit Interactions) ---
    c = CrossLayer(input_shape)(inputs)
    c = CrossLayer(input_shape)(c)
    c = CrossLayer(input_shape)(c)
    
    # --- DEEP NETWORK (Implicit Non-Linearity) ---
    d = layers.Dense(256, activation="swish")(inputs)
    d = layers.BatchNormalization()(d)
    d = layers.Dropout(0.3)(d)
    
    d = layers.Dense(128, activation="swish")(d)
    d = layers.BatchNormalization()(d)
    d = layers.Dropout(0.2)(d)
    
    # --- COMBINE ---
    combined = layers.concatenate([c, d])
    
    outputs = layers.Dense(1, activation="sigmoid")(combined)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    X, y, X_test, ids = load_and_preprocess()
    
    print(f"\n[INFO] Training DCN (Deep & Cross Network)...")
    model = build_dcn_model(X.shape[1])
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=15, 
        restore_best_weights=True,
        mode='max'
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6
    )
    
    history = model.fit(
        X, y,
        validation_split=0.15,
        batch_size=256,
        epochs=120,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("\n[INFO] Generating DCN Predictions...")
    probs = model.predict(X_test)
    preds = (probs > 0.5).astype(int).flatten()
    
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    
    best_val = max(history.history['val_accuracy'])
    print(f"\n[VICTORY] Best Val Acc: {best_val:.5f}")
    print(f"Saved {SUBMISSION_FILE}")

if __name__ == "__main__":
    main()