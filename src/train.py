"""
Primary execution pipeline for the Shinkansen CatBoost architecture.
Orchestrates data ingestion, cross-validation, and SHAP interpretability.
"""
import argparse
import logging
import yaml
from src.data import load_data
from src.model import train_cross_validated_model
from src.evaluate import generate_shap_explanations, generate_learning_curves

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Parses the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_training_pipeline(config_path: str, debug: bool = False):
    """Executes the end-to-end ML pipeline."""
    logger.info(f"Initializing pipeline with configuration: {config_path}")
    config = load_config(config_path)
    
    train_df, test_df = load_data(config_path)
    
    # NOTE: Ensure your src/features.py logic is imported and executed here
    # Example: train_df, test_df = apply_feature_engineering(train_df, test_df)
    
    # Define targets and drop identifiers
    y = train_df['Target']
    X = train_df.drop(columns=['Target', 'ID'])
    
    cat_features = list(X.select_dtypes(include=['object', 'category']).columns)
    
    logger.info("Initiating model training sequence.")
    best_model = train_cross_validated_model(X, y, cat_features, config)
    
    logger.info("Executing SHAP interpretability analysis.")
    generate_shap_explanations(best_model, X)
    
    logger.info("Serializing training telemetry.")
    generate_learning_curves(best_model)
    
    logger.info("Pipeline execution finalized successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Shinkansen CatBoost architecture.")
    parser.add_argument('--config', type=str, default='configs/base.yaml', help="Path to the YAML configuration profile.")
    parser.add_argument('--debug', action='store_true', help="Execute pipeline in debug mode with elevated log verbosity.")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    run_training_pipeline(args.config, args.debug)