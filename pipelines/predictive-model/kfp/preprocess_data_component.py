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
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    import joblib
    import glob

    # Functions from your script
    def load_dataset(file_path):
        df = pd.read_csv(file_path, encoding='latin1')
        print(f"Dataset loaded successfully from {file_path}")
        return df

    def handle_missing_values(df):
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        print("Missing values handled.")
        return df

    def select_features(df):
        selected_columns = [
            'proto', 'service', 'state', 'sbytes', 'dbytes', 'sttl', 'dttl',
            'sloss', 'dloss', 'sload', 'dload', 'spkts', 'dpkts', 'attack_cat', 'label'
        ]
        df = df[selected_columns]
        print("Features selected.")
        return df

    def encode_features(df):
        X = df.drop(['label', 'attack_cat'], axis=1)
        y = df['label']
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
        numerical_transformer = StandardScaler()
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_cols),
                ('num', numerical_transformer, numerical_cols)
            ])
        X_transformed = preprocessor.fit_transform(X)
        print("Features encoded and scaled.")
        return X_transformed, y, preprocessor

    # Main processing
    # List all CSV files
    csv_files = glob.glob(os.path.join(raw_data_path, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_path}")

    # Print available CSV files for debugging
    print(f"Available CSV files: {csv_files}")

    # Try to find the training set file
    training_file = None
    for file in csv_files:
        if 'training' in os.path.basename(file).lower():
            training_file = file
            break

    if training_file is None:
        raise FileNotFoundError("Training set CSV file not found in the provided data path.")

    file_path = training_file

    df = load_dataset(file_path)
    df = handle_missing_values(df)
    df = select_features(df)
    X, y, preprocessor = encode_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save processed data
    os.makedirs(processed_data_path, exist_ok=True)
    joblib.dump(X_train, os.path.join(processed_data_path, 'X_train.pkl'))
    joblib.dump(X_test, os.path.join(processed_data_path, 'X_test.pkl'))
    joblib.dump(y_train, os.path.join(processed_data_path, 'y_train.pkl'))
    joblib.dump(y_test, os.path.join(processed_data_path, 'y_test.pkl'))
    joblib.dump(preprocessor, os.path.join(processed_data_path, 'preprocessor.pkl'))

    print(f"Processed data saved to {processed_data_path}")
