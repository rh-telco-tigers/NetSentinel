# scripts/utils.py

import os
import joblib
import pandas as pd

def load_dataset(file_path):
    """
    Loads a dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def save_object(obj, file_path):
    """
    Saves an object to disk using joblib.

    Args:
        obj: The object to save.
        file_path (str): Path where the object will be saved.
    """
    joblib.dump(obj, file_path)
    print(f"Object saved to {file_path}")

def load_object(file_path):
    """
    Loads an object from disk using joblib.

    Args:
        file_path (str): Path to the saved object.

    Returns:
        The loaded object.
    """
    obj = joblib.load(file_path)
    print(f"Object loaded from {file_path}")
    return obj

def get_feature_names(preprocessor):
    """
    Extracts feature names from the preprocessor.

    Args:
        preprocessor: The fitted ColumnTransformer.

    Returns:
        List[str]: Feature names after preprocessing.
    """
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == 'cat':
            # For categorical features, get the feature names from OneHotEncoder
            ohe = transformer
            ohe_feature_names = ohe.get_feature_names_out(columns)
            feature_names.extend(ohe_feature_names)
        elif name == 'num':
            # For numerical features, names remain the same
            feature_names.extend(columns)
    return feature_names

def set_pandas_display_options():
    """
    Sets display options for pandas DataFrames.
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
