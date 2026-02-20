"""
Primary execution script implementing Stratified K-Fold CV and CatBoost serialization.
"""
import os
import logging
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from .config import CB_PARAMS, MODEL_DIR, SUBMISSION_FILE, RANDOM_SEED
from .data_loader import load_raw_data
from .features import apply_feature_engineering

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training_pipeline():
    """
    Orchestrates data ingestion, feature engineering, cross-validation, and inference.
    """
    train_raw, test_raw = load_raw_data()
    X, y, X_test, test_ids, cat_indices = apply_feature_engineering(train_raw, test_raw)

    os.makedirs(MODEL_DIR, exist_ok=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    test_preds = np.zeros(len(X_test))
    test_pool = Pool(X_test, cat_features=cat_indices)

    logger.info("Initiating Stratified CV phase.")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Executing Fold {fold + 1}")

        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        train_pool = Pool(X_tr, label=y_tr, cat_features=cat_indices)
        val_pool = Pool(X_val, label=y_val, cat_features=cat_indices)

        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=150,
            use_best_model=True
        )

        model_path = os.path.join(MODEL_DIR, f'catboost_fold_{fold}.cbm')
        model.save_model(model_path)

        test_preds += model.predict_proba(test_pool)[:, 1] / skf.n_splits

    final_predictions = (test_preds > 0.5).astype(int)
    sub = pd.DataFrame({'ID': test_ids, 'Overall_Experience': final_predictions})
    sub.to_csv(SUBMISSION_FILE, index=False)
    
    logger.info(f"Inference finalized. Artifacts stored in {MODEL_DIR}.")

if __name__ == "__main__":
    run_training_pipeline()