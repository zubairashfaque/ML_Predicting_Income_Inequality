import yaml
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
import os
import sys
import joblib
import copy
import logging
import plotly.express as px

# Configure logging
log_file = "Project_log.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a function to evaluate a feature set and return the F1 score
# Define a function to evaluate a feature set and return the F1 score
def evaluate_features(X_train, X_test, y_train, y_true, selected_features):
    clf = RandomForestClassifier()  # You can use a different classifier
    clf.fit(X_train.iloc[:, selected_features], y_train)  # Use iloc for integer-based indexing
    y_pred = clf.predict(X_test.iloc[:, selected_features])  # Use iloc for integer-based indexing
    f1 = f1_score(y_true, y_pred, average='weighted')
    return f1

# Define a function to generate and evaluate feature combinations
def generate_and_evaluate_combinations(X_train, X_test, y_train, y_true, column_list):
    best_f1_score = 0
    best_feature_set = None

    for r in range(1, len(column_list) + 1):
        for feature_set in combinations(column_list, r):
            feature_indices = [X_train.columns.get_loc(col) for col in feature_set]
            f1 = evaluate_features(X_train, X_test, y_train, y_true, feature_indices)
            print(f1)
            print(feature_set)
            if f1 > best_f1_score:
                best_f1_score = f1
                best_feature_set = feature_set
            print("BEST FEATURES")
            print(best_f1_score)
            print(best_feature_set)
    return best_feature_set, best_f1_score

# Initialize F1 scores dictionary
f1_scores = {}

# Load data from the YAML file
logging.info("Loading data from YAML file.")
with open("data/final/top_f1_score.yaml", "r") as yaml_file:
    data = yaml.safe_load(yaml_file)

# Extract the list of file names
file_names = data['top_f1_score']

# Separate the file names into training and testing files
test_files = [file.replace("_train.csv", "_test.csv") for file in file_names]
train_files = file_names

# Loop through each train-test file pair
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

    # Step 1: Load the list of columns from the YAML file
    with open("data/processed/column_list.yaml", "r") as yaml_file:
        column_list_data = yaml.safe_load(yaml_file)

    column_list = column_list_data.get("columns", [])

    # Remove 'kfold' and 'income_above_limit' columns
    columns_to_remove = ['kfold', 'income_above_limit']
    column_list = [col for col in column_list if col not in columns_to_remove]

    # Step 2: Generate and evaluate feature combinations
    best_features, accuracy = generate_and_evaluate_combinations(X_train, X_test, y_train, y_true, column_list)

    # Store the results in the f1_scores dictionary
    f1_scores[train_file] = {"Best_Features": best_features, "Accuracy": accuracy}

# Create a color scale for different models
color_scale = px.colors.qualitative.Set1

# Create the output directory if it doesn't exist
output_dir = "data/final"
os.makedirs(output_dir, exist_ok=True)

# Initialize a dictionary to store the result data
result_data = {}

# Loop through each train-test pair and store the best features and accuracy
for train_file, result in f1_scores.items():
    best_features = result["Best_Features"]
    accuracy = result["Accuracy"]
    result_data[train_file] = {"Best_Features": best_features, "Accuracy": accuracy}

    # Create bar plots using Plotly for each train-test pair
    fig = px.bar(x=list(best_features), y=[accuracy] * len(best_features),
                 title=f'F1 Weighted Scores for Classification Models on {train_file}',
                 color=list(best_features), color_continuous_scale=color_scale)
    fig.update_xaxes(categoryorder='total ascending')  # Sort x-axis categories

    # Save the graph as an HTML file
    graph_file_name = f'{train_file.replace("_train.csv", "")}_FE_f1_scores.html'
    graph_file_path = os.path.join(output_dir, graph_file_name)
    fig.write_html(graph_file_path)

    # Show the plot
    fig.show()

# Save the result_data dictionary as a YAML file
with open("data/final/best_features_accuracy.yaml", "w") as yaml_file:
    yaml.dump(result_data, yaml_file)

logging.info("Train files, best features, and accuracy saved in 'best_features_accuracy.yaml'.")
