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
from sklearn.pipeline import Pipeline
from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    SmartCorrelatedSelection,
)

# Load data from the YAML file
with open("data/final/top_f1_score.yaml", "r") as yaml_file:
    data = yaml.safe_load(yaml_file)

# Extract the list of file names
file_names = data['top_f1_score']

# Separate the file names into training and testing files
test_files = [file.replace("_train.csv", "_test.csv") for file in file_names]
train_files = file_names

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
    print(f"Processing train file: {train_file}, test file: {test_file}")

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

    # Create a feature selection pipeline
    pipe = Pipeline([
        ('constant', DropConstantFeatures(tol=0.998)),
        ('duplicated', DropDuplicateFeatures()),
        ('correlation', SmartCorrelatedSelection(selection_method='variance'))
    ])

    # Fit the feature selection pipeline on the training data
    pipe.fit(X_train)

    # Transform the training and test data using the feature selection
    X_train_t = pipe.transform(X_train)
    X_test_t = pipe.transform(X_test)

    # Initialize the Min-Max scaler
    scaler = MinMaxScaler()

    # Fit and transform the scaler on your training data
    X_train = scaler.fit_transform(X_train_t)

    # Transform the test data using the same scaler
    X_test = scaler.transform(X_test_t)

    # Initialize a label encoder
    label_encoder = LabelEncoder()

    # Fit the label encoder on the unique class labels and transform the target variable
    y_train = label_encoder.fit_transform(y_train)
    y_true = label_encoder.transform(y_true)

    # Convert the transformed features back to a Pandas DataFrame
    X_train = pd.DataFrame(X_train, columns=X_train_t.columns)
    X_test = pd.DataFrame(X_test, columns=X_test_t.columns)

    # Apply each classification model
    for model_name, model in models.items():
        model.fit(X_train, y_train)  # Train the model
        y_pred = model.predict(X_test)  # Make predictions on the test data
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_scores[train_file][model_name] = f1_weighted

# Create DataFrames from the F1 scores
f1_dfs = {train_file: pd.DataFrame.from_dict(f1_scores[train_file], orient='index') for train_file in train_files}

# Create a color scale for different models
color_scale = px.colors.qualitative.Set1

# Create the output directory if it doesn't exist
output_dir = "data/final"
os.makedirs(output_dir, exist_ok=True)

# Create bar plots using Plotly for each train-test pair
for train_file, f1_df in f1_dfs.items():
    fig = px.bar(f1_df, x=f1_df.index, y=f1_df[0].tolist(), title=f'F1 Weighted Scores for Classification Models on {train_file}',
                 color=f1_df.index, color_continuous_scale=color_scale)
    fig.update_xaxes(categoryorder='total ascending')  # Sort x-axis categories

    # Save the graph as an HTML file
    graph_file_name = f'{train_file.replace("_train.csv", "")}_FE_f1_scores.html'
    graph_file_path = os.path.join(output_dir, graph_file_name)
    fig.write_html(graph_file_path)

    # Show the plot
    fig.show()

# Create the DataFrame
df_1 = pd.DataFrame.from_dict(f1_scores, orient='index')

# Print the DataFrame
print(df_1)

# Calculate the mean F1 score for each row and add it as a new column
df_1['Mean'] = df_1.mean(axis=1)

# Display the updated DataFrame
print(df_1)

# Save the DataFrame as a CSV file
df_1.to_csv(os.path.join(output_dir, 'f1_scores_tp_features.csv'), index=True)

# Print a message indicating the save is complete
print("All graphs and DataFrame saved in the 'data/final' folder.")