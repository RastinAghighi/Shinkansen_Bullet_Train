"""
Unit tests for the Shinkansen ML pipeline infrastructure.
Executes configuration validation and model initialization checks.
"""
import os
import pytest
import yaml
from catboost import CatBoostClassifier

def test_config_parsing():
    """Validates YAML configuration schema and disk presence."""
    config_path = "configs/base.yaml"
    assert os.path.exists(config_path), "Configuration file missing."
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    assert "model" in config, "Model key missing from configuration."
    assert config["model"] == "catboost", "Incorrect model architecture specified."
    assert "data" in config, "Data routing configuration missing."

def test_model_initialization():
    """Verifies CatBoost object instantiation with override parameters."""
    model = CatBoostClassifier(iterations=10, depth=4, silent=True)
    assert model is not None, "Model failed to initialize."
    assert model.get_param('iterations') == 10, "Parameter override failed."