import yaml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, HistGradientBoostingClassifier
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import os
import logging

# Configure logging
logging.basicConfig(filename='Project_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data from the YAML file
try:
    with open("data/processed/sample_data/top_sample_list.yaml", "r") as yaml_file:
        data = yaml.safe_load(yaml_file)

    # Extract the list of file names
    file_names = data

    # Separate the file names into training and testing files
    train_files = [file.replace("_test.csv", "_train.csv") for file in file_names]
    test_files = file_names

    # Initialize all the classification models
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'GBM': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'GTB': GradientBoostingClassifier(),
        'Extra Trees': ExtraTreesClassifier(),
        'XGBoost': xgb.XGBClassifier(),
        'LightGBM': lgb.LGBMClassifier(verbose=-1),
        'CatBoost': cb.CatBoostClassifier(silent=True),
        'HistGradientBoosting': HistGradientBoostingClassifier()
    }

    # Initialize F1 scores dictionary
    f1_scores = {}

    # Loop through each train and test file and apply the models
    for train_file, test_file in zip(train_files, test_files):
        logging.info(f"Processing train file: {train_file}, test file: {test_file}")

        # Load train and test data
        train_data = pd.read_csv(f'data/processed/sample_data/{train_file}')
        test_data = pd.read_csv(f'data/processed/sample_data/{test_file}')

        # Split the data into features (X) and target (y)
        X_train = train_data.drop('income_above_limit', axis=1)
        y_train = train_data['income_above_limit']

        X_test = test_data.drop(['income_above_limit', 'kfold'], axis=1)
        y_true = test_data['income_above_limit']

        # Initialize F1 scores for this train-test pair
        f1_scores[train_file] = {}

        # Initialize the Min-Max scaler
        scaler = MinMaxScaler()

        # Fit and transform the scaler on your training data
        X_train = scaler.fit_transform(X_train)

        # Transform the test data using the same scaler
        X_test = scaler.transform(X_test)

        # Initialize a label encoder
        label_encoder = LabelEncoder()

        # Fit the label encoder on the unique class labels and transform the target variable
        y_train = label_encoder.fit_transform(y_train)
        y_true = label_encoder.transform(y_true)

        # Apply each classification model
        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)  # Train the model
                y_pred = model.predict(X_test)  # Make predictions on the test data
                f1_weighted = f1_score(y_true, y_pred, average='weighted')
                f1_scores[train_file][model_name] = f1_weighted
                logging.info(f"Processed train file: {train_file}, test file: {test_file}, model: {model_name}, F1 Score: {f1_weighted}")
            except Exception as e:
                logging.error(f"Error processing train file: {train_file}, test file: {test_file}, model: {model_name}: {str(e)}")

    # Create DataFrames from the F1 scores
    f1_dfs = {train_file: pd.DataFrame.from_dict(f1_scores[train_file], orient='index') for train_file in train_files}

    # Create a color scale for different models
    color_scale = px.colors.qualitative.Set1

    # Create the output directory if it doesn't exist
    output_dir = "data/final"
    os.makedirs(output_dir, exist_ok=True)

    # Create bar plots using Plotly for each train-test pair
    for train_file, f1_df in f1_dfs.items():
        try:
            fig = px.bar(f1_df, x=f1_df.index, y=f1_df[0].tolist(), title=f'F1 Weighted Scores for Classification Models on {train_file}',
                         color=f1_df.index, color_continuous_scale=color_scale)
            fig.update_xaxes(categoryorder='total ascending')  # Sort x-axis categories

            # Save the graph as an HTML file
            graph_file_name = f'{train_file.replace("_train.csv", "")}_f1_scores.html'
            graph_file_path = os.path.join(output_dir, graph_file_name)
            fig.write_html(graph_file_path)

            # Show the plot
            fig.show()
            logging.info(f"Saved F1 score graph for train file: {train_file} to {graph_file_path}")
        except Exception as e:
            logging.error(f"Error creating F1 score graph for train file: {train_file}: {str(e)}")

    # Create the DataFrame
    df_1 = pd.DataFrame.from_dict(f1_scores, orient='index')

    # Calculate the mean F1 score for each row and add it as a new column
    df_1['Mean'] = df_1.mean(axis=1)

    # Save the DataFrame as a CSV file
    df_1.to_csv(os.path.join(output_dir, 'f1_scores.csv'), index=True)
    logging.info("Saved F1 scores DataFrame as 'f1_scores.csv'")

    # Select the top four mean files' names
    top_mean_files = df_1.sort_values(by='Mean', ascending=False).index.tolist()[:4]

    # Print the list of top four mean files' names
    logging.info("Top Four Mean Files Names:")
    logging.info(top_mean_files)

    # Define the output YAML file path
    output_yaml_path = "data/final/top_f1_score.yaml"

    # Create a dictionary to store the top_f1_score
    top_f1_score_dict = {"top_f1_score": top_mean_files}

    # Save the dictionary to the YAML file
    with open(output_yaml_path, "w") as yaml_file:
        yaml.dump(top_f1_score_dict, yaml_file)
    logging.info(f"Saved top four mean files' names in '{output_yaml_path}'")

    logging.info("All graphs, DataFrame, and YAML file saved successfully.")
except Exception as e:
    logging.error(f"Error: {str(e)}")
