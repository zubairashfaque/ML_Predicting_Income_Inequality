import yaml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
from sklearn.model_selection import train_test_split
import os
import logging
import plotly.express as px

# Configure logging
log_file = "Project_log.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data from the YAML file
logging.info("Loading data from YAML file.")
with open("data/final/top_f1_score.yaml", "r") as yaml_file:
    data = yaml.safe_load(yaml_file)

# Extract the list of file names
file_names = data['top_f1_score']

# Separate the file names into training and testing files
test_files = [file.replace("_train.csv", "_test.csv") for file in file_names]
train_files = file_names

# Define a function to evaluate a feature set and return the F1 score
def evaluate_features(X_train, X_test, y_train, y_true, selected_features):
    clf = RandomForestClassifier()  # You can use a different classifier
    clf.fit(X_train[:, selected_features], y_train)
    y_pred = clf.predict(X_test[:, selected_features])
    f1 = f1_score(y_true, y_pred, average='weighted')
    return f1

# Initialize F1 scores dictionary
f1_scores = {}

# Loop through each train and test file and apply the models
for train_file, test_file in zip(train_files, test_files):
    logging.info(f"Processing train file: {train_file}, test file: {test_file}")

    # Load train and test data
    train_data = pd.read_csv(f'data/processed/sample_data/{train_file}')  # Adjust the file path as needed
    test_data = pd.read_csv(f'data/processed/sample_data/{test_file}')  # Adjust the file path as needed

    # Split the data into features (X) and target (y)
    X_train = train_data.drop('income_above_limit', axis=1)
    y_train = train_data['income_above_limit']

    X_test = test_data.drop(['income_above_limit', 'kfold'], axis=1)
    y_true = test_data['income_above_limit']

    # Initialize F1 scores for this train-test pair
    f1_scores[train_file] = {}

    # Create a list of all possible feature combinations
    all_feature_combinations = []
    for r in range(1, len(X_train.columns) + 1):
        all_feature_combinations.extend(combinations(range(len(X_train.columns)), r))

    best_f1_score = 0
    best_feature_set = None
    print(all_feature_combinations)
    # Iterate through all feature combinations and find the best one
    for feature_set in all_feature_combinations:
        f1 = evaluate_features(X_train, X_test, y_train, y_true, feature_set)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_feature_set = feature_set
            print(best_f1_score)

    f1_scores[train_file]["Best"] = best_f1_score

# Create DataFrames from the F1 scores
f1_df = pd.DataFrame.from_dict(f1_scores, orient='index')

# Create a color scale for different models
color_scale = px.colors.qualitative.Set1

# Create the output directory if it doesn't exist
output_dir = "data/final"
os.makedirs(output_dir, exist_ok=True)

# Create bar plots using Plotly for each train-test pair
for train_file, f1_score in f1_df.iterrows():
    fig = px.bar(f1_score, x=f1_score.index, y=f1_score, title=f'F1 Weighted Scores for Classification Models on {train_file}',
                 color=f1_score.index, color_continuous_scale=color_scale)
    fig.update_xaxes(categoryorder='total ascending')  # Sort x-axis categories

    # Save the graph as an HTML file
    graph_file_name = f'{train_file.replace("_train.csv", "")}_FE_f1_scores.html'
    graph_file_path = os.path.join(output_dir, graph_file_name)
    fig.write_html(graph_file_path)

    # Show the plot
    fig.show()

# Initialize a dictionary to store the data
result_data = {}

# Loop through each train-test pair and store the best features and accuracy
for train_file, best_f1_score in f1_df.iterrows():
    # Get the best features
    best_features = train_data.columns[best_features.index("Best")]

    # Get the best accuracy
    accuracy = best_f1_score["Best"]

    # Store the data in the dictionary
    result_data[train_file] = {"Best_Features": best_features, "Accuracy": accuracy}

# Save the result_data dictionary as a YAML file
with open("data/final/best_features_accuracy.yaml", "w") as yaml_file:
    yaml.dump(result_data, yaml_file)

logging.info("Train files, best features, and accuracy saved in 'best_features_accuracy.yaml'.")
