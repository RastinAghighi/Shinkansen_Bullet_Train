"""
Model evaluation and interpretability module.
Generates performance metrics and SHAP value aggregations for CatBoost architectures.
"""
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

def generate_shap_explanations(model: CatBoostClassifier, X_val: pd.DataFrame, output_dir: str = "results") -> str:
    """
    Calculates SHAP values and exports a summary plot for feature importance analysis.
    
    Args:
        model (CatBoostClassifier): Trained CatBoost model instance.
        X_val (pd.DataFrame): Validation feature matrix.
        output_dir (str): Destination directory for generated artifacts.
        
    Returns:
        str: Path to the saved SHAP summary plot.
    """
    logger.info("Initiating exact SHAP value calculation via TreeExplainer.")
    
    # TreeExplainer provides O(TLD^2) exact calculation natively optimized for ensemble trees
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val, show=False)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "shap_summary.png")
    
    # bbox_inches prevents axis label truncation during headless matplotlib rendering
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logger.info(f"SHAP summary plot serialized to {out_path}")
    return out_path


def generate_learning_curves(model: CatBoostClassifier, output_dir: str = "results") -> str:
    """
    Extracts evaluation telemetry from the model and serializes learning curve artifacts.
    
    Args:
        model (CatBoostClassifier): Trained CatBoost model instance.
        output_dir (str): Destination directory for generated artifacts.
        
    Returns:
        str: Path to the saved learning curves plot.
    """
    logger.info("Generating learning curve artifacts from model telemetry.")
    evals_result = model.evals_result_
    
    if not evals_result:
        logger.warning("No evaluation metrics found in model telemetry.")
        return ""
        
    plt.figure(figsize=(10, 6))
    
    for dataset_name, metrics in evals_result.items():
        for metric_name, values in metrics.items():
            plt.plot(values, label=f"{dataset_name} - {metric_name}")
            
    plt.title("Model Convergence: Learning Curves")
    plt.xlabel("Iterations")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "learning_curves.png")
    
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logger.info(f"Learning curve artifact serialized to {out_path}")
    return out_path