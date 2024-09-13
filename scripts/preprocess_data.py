# scripts/preprocess_data.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def load_dataset(file_path):
    """
    Loads the UNSW-NB15 dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def handle_missing_values(df):
    """
    Handles missing values in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    # Fill missing values for categorical features with 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    # Fill missing values for numerical features with the median
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    print("Missing values handled.")
    return df

def select_features(df):
    """
    Selects relevant features for intrusion detection.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with selected features.
    """
    selected_columns = [
        'proto', 'service', 'state', 'sbytes', 'dbytes', 'sttl', 'dttl',
        'sloss', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'attack_cat', 'label'
    ]
    df = df[selected_columns]
    print("Features selected.")
    return df

def encode_features(df):
    """
    Encodes categorical variables and scales numerical features.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        np.ndarray: Transformed feature array.
        np.ndarray: Label array.
        ColumnTransformer: Fitted transformer for future use.
    """
    # Separate features and labels
    X = df.drop(['label', 'attack_cat'], axis=1)
    y = df['label']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Define transformers
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
    numerical_transformer = StandardScaler()

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ])

    # Fit and transform the data
    X_transformed = preprocessor.fit_transform(X)

    print("Features encoded and scaled.")
    return X_transformed, y.values, preprocessor

def save_processed_data(X_train, X_test, y_train, y_test, preprocessor, output_dir):
    """
    Saves the processed data and preprocessor for future use.

    Args:
        X_train, X_test, y_train, y_test: Training and testing data.
        preprocessor (ColumnTransformer): Fitted preprocessor.
        output_dir (str): Directory to save processed data.
    """
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(X_train, os.path.join(output_dir, 'X_train.pkl'))
    joblib.dump(X_test, os.path.join(output_dir, 'X_test.pkl'))
    joblib.dump(y_train, os.path.join(output_dir, 'y_train.pkl'))
    joblib.dump(y_test, os.path.join(output_dir, 'y_test.pkl'))
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))

    print(f"Processed data saved to {output_dir}")

if __name__ == "__main__":
    # Paths
    DATA_DIR = os.path.join('data', 'raw')
    OUTPUT_DIR = os.path.join('data', 'processed')
    DATA_FILE = os.path.join(DATA_DIR, 'UNSW_NB15_training-set.csv')

    # Load dataset
    df = load_dataset(DATA_FILE)
    if df is not None:
        # Handle missing values
        df = handle_missing_values(df)
        # Select features
        df = select_features(df)
        # Encode features and labels
        X, y, preprocessor = encode_features(df)
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Data split into training and testing sets.")
        # Save processed data
        save_processed_data(X_train, X_test, y_train, y_test, preprocessor, OUTPUT_DIR)
    else:
        print("Data preprocessing failed due to dataset loading error.")
