"""
Global configuration and hyperparameter definitions for the Shinkansen model pipeline.
"""
import os

ZIP_PATH = os.path.join('data', 'Olympus', 'archive.zip')
MODEL_DIR = 'models'
SUBMISSION_FILE = 'Submission_Native_Cat_CV.csv'
RANDOM_SEED = 42

FILES = {
    'train_travel': 'Traveldata_train_(1).csv',
    'train_survey': 'Surveydata_train_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

CB_PARAMS = {
    'iterations': 3500,
    'depth': 8,
    'learning_rate': 0.05,
    'l2_leaf_reg': 1.96,
    'border_count': 152,
    'loss_function': 'Logloss',
    'verbose': 500,
    'random_seed': RANDOM_SEED
}