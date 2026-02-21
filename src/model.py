"""
Core modeling module.
Implements CatBoost architecture with Stratified K-Fold CV, early stopping, and automated checkpointing.
"""
import logging
from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

def train_cross_validated_model(X: pd.DataFrame, y: pd.Series, cat_features: list, config: dict, output_dir: str = "results") -> CatBoostClassifier:
    """
    Executes Stratified K-Fold cross-validation with early stopping and model checkpointing.
    
    Args:
        X (pd.DataFrame): Training feature matrix.
        y (pd.Series): Target vector.
        cat_features (list): Indices of categorical features.
        config (dict): Hyperparameter dictionary.
        output_dir (str): Directory for model checkpointing.
        
    Returns:
        CatBoostClassifier: The model trained on the highest-performing fold.
    """
    cv_folds = config.get('cv_folds', 5)
    seed = config.get('random_seed', 42)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    
    best_model = None
    best_score = 0.0
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info("Initializing Fold %d/%d", fold + 1, cv_folds)
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Centralizing data and categorical metadata via Pool avoids serialization 
        # mismatches during cross-validation boundary splits.
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        
        model = CatBoostClassifier(
            iterations=config.get('iterations', 3500),
            learning_rate=config.get('learning_rate', 0.05),
            depth=config.get('depth', 8),
            l2_leaf_reg=config.get('l2_leaf_reg', 1.96),
            border_count=config.get('border_count', 152),
            random_seed=seed,
            eval_metric='Accuracy',
            early_stopping_rounds=config.get('early_stopping_rounds', 150),
            verbose=False
        )
        
        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True
        )
        
        fold_accuracy = model.best_score_['validation']['Accuracy']
        logger.info("Fold %d complete. Validation Accuracy: %.4f", fold + 1, fold_accuracy)
        
        if fold_accuracy > best_score:
            best_score = fold_accuracy
            best_model = model
            
            checkpoint_path = Path(output_dir) / "best_catboost_model.cbm"
            best_model.save_model(str(checkpoint_path))
            
    logger.info("Cross-validation finalized. Peak Validation Accuracy: %.4f", best_score)
    logger.info("Optimal model checkpointed at %s", checkpoint_path)
    return best_model