"""
Primary execution pipeline for the Shinkansen CatBoost architecture.
Orchestrates data ingestion, cross-validation, and SHAP interpretability.
"""
import argparse
import logging
import yaml
from src.data import load_data
from src.features import apply_feature_engineering
from src.model import train_cross_validated_model
from src.evaluate import generate_shap_explanations, generate_learning_curves

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_training_pipeline(config: dict, output_dir: str, debug: bool = False):
    """Executes the end-to-end ML pipeline."""
    logger.info("Initializing pipeline execution.")
    train_df, test_df = load_data(config)
    
    X, y, X_test, test_ids, cat_indices = apply_feature_engineering(train_df, test_df)
    
    logger.info("Initiating model training sequence.")
    best_model = train_cross_validated_model(X, y, cat_indices, config, output_dir=output_dir)
    
    logger.info("Executing SHAP interpretability analysis.")
    generate_shap_explanations(best_model, X, output_dir=output_dir)
    
    logger.info("Serializing training telemetry.")
    generate_learning_curves(best_model, output_dir=output_dir)
    
    logger.info("Pipeline execution finalized successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Shinkansen CatBoost architecture.")
    parser.add_argument('--config', type=str, default='configs/base.yaml', help="Path to the YAML configuration profile.")
    parser.add_argument('--data_dir', type=str, help="Override path to the dataset directory.")
    parser.add_argument('--iterations', type=int, help="Override the number of boosting iterations.")
    parser.add_argument('--output_dir', type=str, default='results', help="Directory to save models and artifacts.")
    parser.add_argument('--debug', action='store_true', help="Execute pipeline in debug mode with elevated log verbosity.")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    with open(args.config, 'r') as file:
        runtime_config = yaml.safe_load(file)
        
    if args.iterations:
        runtime_config['iterations'] = args.iterations
    if args.data_dir:
        runtime_config['data']['train_path'] = f"{args.data_dir}/train.csv"
        runtime_config['data']['test_path'] = f"{args.data_dir}/test.csv"
        
    run_training_pipeline(runtime_config, args.output_dir, args.debug)