# n02_preprocess_data_component.py

import kfp
from kfp import dsl
from kfp.dsl import component, InputPath, OutputPath

@component(
    packages_to_install=['pandas', 'numpy', 'scikit-learn', 'joblib'],
)
def preprocess_data_component(
    raw_data_path: InputPath(),
    processed_data_path: OutputPath(),
):
    import os
    import pandas as pd
    import numpy as np
    import glob
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import joblib

    # List files in raw_data_path
    print(f"Files in raw_data_path: {os.listdir(raw_data_path)}")

    # Read the features file to get the column names
    features_file = os.path.join(raw_data_path, "NUSW-NB15_features.csv")
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found at {features_file}")
    features = pd.read_csv(features_file, encoding='cp1252')
    feature_names = features['Name'].tolist()

    # Read only UNSW-NB15_1.csv to UNSW-NB15_4.csv
    csv_files = [os.path.join(raw_data_path, f"UNSW-NB15_{i}.csv") for i in range(1, 5)]
    dataframes = []
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Data file not found: {csv_file}")
        print(f"Reading {csv_file}")
        df = pd.read_csv(csv_file, header=None, encoding='latin1', low_memory=False)
        if df.shape[1] != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} columns, but got {df.shape[1]} columns in {csv_file}")
        df.columns = feature_names
        dataframes.append(df)
        print(f"Loaded {csv_file} with shape {df.shape}")

    # Concatenate dataframes
    train_df = pd.concat(dataframes, ignore_index=True)
    print(f"Dataset shape after concatenation: {train_df.shape}")

    # Shuffle the data
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Remove duplicates
    train_df = train_df.drop_duplicates()
    print(f"Dataset shape after removing duplicates: {train_df.shape}")

    # Handle missing values
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    train_df[categorical_cols] = train_df[categorical_cols].fillna('Unknown')

    numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    train_df[numerical_cols] = train_df[numerical_cols].fillna(train_df[numerical_cols].median())
    print("Missing values handled.")

    # Encode categorical variables using LabelEncoder
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        label_encoders[col] = le
    print(f"Categorical variables encoded: {list(categorical_cols)}")

    # Separate features and target
    X = train_df.drop(['Label', 'attack_cat'], axis=1)
    y = train_df['attack_cat']

    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print("Target variable encoded.")

    # Scale numerical features
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    print("Numerical features scaled.")

    # Save preprocessed data and encoders
    os.makedirs(processed_data_path, exist_ok=True)
    X.to_pickle(os.path.join(processed_data_path, 'X.pkl'))
    joblib.dump(y_encoded, os.path.join(processed_data_path, 'y_encoded.pkl'))
    joblib.dump(label_encoders, os.path.join(processed_data_path, 'label_encoders.joblib'))
    joblib.dump(scaler, os.path.join(processed_data_path, 'scaler.joblib'))
    joblib.dump(label_encoder, os.path.join(processed_data_path, 'label_encoder.joblib'))
    print(f"Processed data saved to {processed_data_path}")
