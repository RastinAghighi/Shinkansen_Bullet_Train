"""
Unit tests for the feature engineering pipeline.
"""
import unittest
import pandas as pd
from src.features import apply_feature_engineering

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        """
        Creates a synthetic, minimal dataset mimicking the Shinkansen survey structure.
        """
        self.train_mock = pd.DataFrame({
            'ID': [1, 2],
            'Overall_Experience': [1, 0],
            'Departure_Delay_in_Mins': [10.0, None],
            'Arrival_Delay_in_Mins': [15.0, 5.0],
            'Seat_Comfort': ['Excellent', None] # Intentionally leaving one blank
        })
        
        self.test_mock = pd.DataFrame({
            'ID': [3],
            'Departure_Delay_in_Mins': [0.0],
            'Arrival_Delay_in_Mins': [0.0],
            'Seat_Comfort': ['Good']
        })

    def test_missing_data_imputation(self):
        """
        Validates that NaN values in categorical object columns are explicitly 
        cast to the 'Missing_Data' string to preserve behavioral non-response.
        """
        X, y, X_test, test_ids, cat_indices = apply_feature_engineering(self.train_mock, self.test_mock)
        
        # Check if the NaN in 'Seat_Comfort' was correctly mapped
        imputed_value = X.loc[1, 'Seat_Comfort']
        self.assertEqual(imputed_value, 'Missing_Data', "Categorical NaN was not mapped to 'Missing_Data'")
        
        # Ensure the target vector matches the length of the training matrix
        self.assertEqual(len(X), len(y), "Target vector length mismatch")

if __name__ == '__main__':
    unittest.main()