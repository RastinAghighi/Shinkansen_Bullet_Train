"""
Feature engineering and behavioral imputation module.
"""
import pandas as pd

def apply_feature_engineering(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[int]]:
    """
    Executes delay calculations and categorical imputation mapping.
    """
    target = train['Overall_Experience']
    test_ids = test['ID']

    train.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test.drop(['ID'], axis=1, inplace=True)

    df = pd.concat([train, test], axis=0).reset_index(drop=True)

    df['Total_Delay'] = df['Departure_Delay_in_Mins'].fillna(0) + df['Arrival_Delay_in_Mins'].fillna(0)
    df['Delay_Ratio'] = df['Arrival_Delay_in_Mins'] / (df['Departure_Delay_in_Mins'] + 1)

    cat_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Isolate non-response as a distinct decision boundary
            df[col] = df[col].fillna("Missing_Data").astype(str)
            cat_cols.append(col)

    X = df.iloc[:len(train)].copy()
    X_test = df.iloc[len(train):].copy()

    cat_indices = [X.columns.get_loc(c) for c in cat_cols]

    return X, target, X_test, test_ids, cat_indices