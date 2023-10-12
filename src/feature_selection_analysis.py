import os
import yaml
import pandas as pd
import plotly.express as px
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, HistGradientBoostingClassifier
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    SmartCorrelatedSelection,
)

# Initialize all the classification models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'GBM': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'GTB': GradientBoostingClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'LightGBM': lgb.LGBMClassifier(verbose=-1),  # Suppress LightGBM verbose output
    'CatBoost': cb.CatBoostClassifier(silent=True),  # Suppress CatBoost verbose output
}

# Load data from the YAML file
with open("data/final/top_f1_score.yaml", "r") as yaml_file:
    data = yaml.safe_load(yaml_file)

# Extract the list of file names
file_names = data['top_f1_score']

# Directory to save feature importances
feature_importance_dir = "data/final/feature_importances"

# Initialize F1 scores dictionary
f1_scores = {}

# Initialize an empty list to store results
results = []

# Loop through each train file (feature importance file) and apply the models
for train_file in file_names:
    for model_name, model in models.items():
        # Load the feature importance file for this model
        feature_importance_file = os.path.join(feature_importance_dir,
                                               f'{train_file.replace("_train.csv", "")}_{model_name}_top_10_features.csv')
        feature_importance_df = pd.read_csv(feature_importance_file)

        # Load the corresponding train data file
        train_data_file = os.path.join('data/processed/sample_data', train_file)
        train_data = pd.read_csv(train_data_file)

        # Extract the top 10 features
        top_10_features = feature_importance_df['Feature'][:10].tolist()

        # Split the data into features (X) and target (y)
        X_train = train_data[top_10_features]
        y_train = train_data['income_above_limit']

        # Initialize the Min-Max scaler
        scaler = MinMaxScaler()

        # Fit and transform the scaler on your training data
        X_train = scaler.fit_transform(X_train)

        # Initialize a label encoder
        label_encoder = LabelEncoder()

        # Fit the label encoder on the unique class labels and transform the target variable
        y_train = label_encoder.fit_transform(y_train)

        # Apply the model
        model.fit(X_train, y_train)  # Train the model on the top 10 features

        # Load the test data
        test_file = train_file.replace("_train.csv", "_test.csv")
        test_data_file = os.path.join('data/processed/sample_data', test_file)
        test_data = pd.read_csv(test_data_file)

        # Extract the top 10 features from the test data
        X_test = test_data[top_10_features]
        y_true = test_data['income_above_limit']

        # Transform the test data using the same scaler
        X_test = scaler.transform(X_test)

        # Transform the target variable
        y_true = label_encoder.transform(y_true)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate F1 weighted score
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        # Store the F1 score in the dictionary
        if train_file not in f1_scores:
            f1_scores[train_file] = {}
        f1_scores[train_file][model_name] = f1_weighted

        # Append the results to the list
        results.append({
            'Filename': train_file,
            'Model': model_name,
            'F1 Weighted Score': f1_weighted
        })

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Save the results DataFrame to a CSV file in the feature_importance_dir
results_file_path = os.path.join(feature_importance_dir, 'results.csv')
results_df.to_csv(results_file_path, index=False)
print(results_df)

# Create DataFrames from the F1 scores
f1_dfs = {train_file: pd.DataFrame.from_dict(f1_scores[train_file], orient='index') for train_file in file_names}

# Create a color scale for different models
color_scale = px.colors.qualitative.Set1

# Create bar plots using Plotly for each train-test pair with color scale
for train_file, f1_df in f1_dfs.items():
    fig = px.bar(f1_df, x=f1_df.index, y=f1_df[0].tolist(), title=f'F1 Weighted Scores for Classification Models on {train_file}',
                 color=f1_df.index, color_continuous_scale=color_scale)
    fig.update_xaxes(categoryorder='total ascending')  # Sort x-axis categories

    # Save the graph as an HTML file
    graph_file_name = f'{train_file.replace("_train.csv", "")}_FS_f1_scores.html'
    graph_file_path = os.path.join("data/final", graph_file_name)
    fig.write_html(graph_file_path)

    # Show the plot
    fig.show()

# Print a message indicating the script has completed execution
print("Script execution completed.")