import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
import os
import sys
import joblib
import copy
import logging
import yaml

# Configure logging
logging.basicConfig(filename='Project_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def drop_columns_with_missing_values(df, threshold=0.4):
    """
    Drops columns from a DataFrame if they have more than a specified threshold of missing values.

    Args:
    df (pd.DataFrame): The input DataFrame.
    threshold (float): The threshold for the proportion of missing values. Columns with missing values exceeding this proportion will be dropped.

    Returns:
    pd.DataFrame: The DataFrame with columns having more than the threshold of missing values dropped.
    """
    try:
        # Calculate the proportion of missing values for each column
        missing_proportion = df.isnull().mean()

        # Identify columns that exceed the threshold
        columns_to_drop = missing_proportion[missing_proportion > threshold].index.tolist()

        # Drop the identified columns
        df_cleaned = df.drop(columns=columns_to_drop)

        # Print information about deleted and remaining columns
        deleted_columns_count = len(columns_to_drop)
        deleted_columns_names = ", ".join(columns_to_drop)
        remaining_columns_count = df_cleaned.shape[1]

        print(f"Deleted {deleted_columns_count} columns: {deleted_columns_names}")
        print(f"Remaining {remaining_columns_count} columns")

        return df_cleaned
    except Exception as e:
        logging.error(f"Error in drop_columns_with_missing_values: {str(e)}")


def preprocess_data(input_file, output_file, ordinal_encoder_mapping_file):
    """
    Preprocesses a CSV file containing RTA data.

    This function reads a CSV file containing RTA data, performs various data preprocessing steps, and saves the
    processed data to a new CSV file.

    Args:
        input_file (str): The path to the input CSV file containing the raw RTA data.
        output_file (str): The path to the output CSV file where the processed data will be saved.
        ordinal_encoder_mapping_file (str): The path to the JSON file where ordinal encoder mappings will be saved.

    Returns:
        None
    """
    try:
        logging.info(
            f"-----------------------------------------Data preprocessing Started-----------------------------------------")
        # Step 1: Read the CSV file into a DataFrame
        df = pd.read_csv(input_file)
        print("Step 1: CSV file read into DataFrame.")
        logging.info(
            f"-----------------------------------------Data Loaded-----------------------------------------")
        # Step 2: Drop unnecessary columns
        df = drop_columns_with_missing_values(df, threshold=0.4)
        logging.info(
            f"-----------------------------------------NAN Columns Dropped-----------------------------------------")
        # Define the target column
        target = 'income_above_limit'
        #delete column with column name 'ID'
        df = df.drop(columns=['ID'])
        # Step 3: Split the data into folds using StratifiedKFold
        kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df[target].values)):
            df.loc[val_idx, "kfold"] = fold
        logging.info(
            f"-----------------------------------------Split the data into folds using StratifiedKFold done-----------------------------------------")
        columns_to_convert = ['vet_benefit', 'mig_year', 'total_employed', 'employment_stat']

        df[columns_to_convert] = df[columns_to_convert].astype('object')

        df_deep_copy = copy.deepcopy(df)

        # Step 7: Perform ordinal encoding on categorical columns (excluding target)
        encoder = OrdinalEncoder()
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
        ordinal_encoders = {}  # Dictionary to store ordinal encoder mappings

        joblib.dump(encoder, ordinal_encoder_mapping_file)

        # Save the ordinal encoder for specific columns to a separate file
        columns_to_encode = [
            'education',
            'gender',
            'industry_code',
            'occupation_code',
            'industry_code',
            'tax_status',
            'industry_code_main'
        ]
        df_deep_copy[columns_to_encode] = encoder.fit_transform(df_deep_copy[columns_to_encode])
        joblib.dump(encoder, 'data/processed/ordinal_encoder_final.pkl')
        logging.info(
            f"-----------------------------------------Ordinal Encoding Done-----------------------------------------")
        # Step 13: Define the number of neighbors for KNN imputation
        k_neighbors = 5

        # Initialize the KNN imputer
        imputer = KNNImputer(n_neighbors=k_neighbors)

        # Step 9: Perform KNN imputation on the dataset
        imputed_data = imputer.fit_transform(df)
        print("Step 2: Missing values imputed using KNNImputer.")
        logging.info(
            f"-----------------------------------------KNN imputation Done-----------------------------------------")
        # Convert the imputed data back to a DataFrame
        df = pd.DataFrame(imputed_data, columns=df.columns)

        # Step 10: Save the processed DataFrame to the output CSV file
        df.to_csv(output_file, index=False)
        print(f"Step 3: Processed data saved to {output_file}.")

        logging.info(
            f"-----------------------------------------Data preprocessing completed successfully-----------------------------------------")
         # Step 12: Save the list of columns in a YAML file
        column_list = df.columns.tolist()
        with open('data/processed/column_list.yaml', "w") as yaml_file:
            yaml.dump({"columns": column_list}, yaml_file)
        print(f"Step 4: List of columns saved to {'data/processed/column_list.yaml'}.")

    except Exception as e:
        logging.error(f"Error in preprocess_data: {str(e)}")


# Example usage:
if __name__ == "__main__":
    try:
        if len(sys.argv) != 3 and len(sys.argv) != 5:
            raise ValueError("Invalid number of arguments. Usage: python script.py input_dir_path output_dir_path")

        input_dir = sys.argv[1]
        output_dir = sys.argv[2]

        train_input_cv = os.path.join(input_dir, "data.csv")
        train_output_cv = os.path.join(output_dir, "FE_output.csv")

        ordinal_encoder_mapping_file = os.path.join(output_dir, "ordinal_encoder_mapping.json")

        preprocess_data(train_input_cv, train_output_cv, ordinal_encoder_mapping_file)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
