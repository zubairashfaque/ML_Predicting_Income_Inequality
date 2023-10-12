import os
import yaml
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import plotly.express as px
from collections import OrderedDict
from joblib import dump

# Step 1: Define a function to train and evaluate a model
def train_and_evaluate_model(filename, hyperparameters):
    print(f"Step 1: Loading feature importance file for {filename}...")
    # Step 2: Load the feature importance file for this model
    new_filename = filename.replace("_train.csv", "_LightGBM_top_10_features.csv")
    file_path = os.path.join(feature_importance_dir, new_filename)
    feature_importance_df = pd.read_csv(file_path)

    print(f"Step 2: Loading train data file {filename}...")
    # Step 3: Load the corresponding train data file
    train_data_file = os.path.join(sample_dir, filename)
    train_data = pd.read_csv(train_data_file)

    print(f"Step 3: Splitting data into features and target for {filename}...")
    # Step 4: Split the data into features (X) and target (y)
    X_train = train_data.drop(columns=['income_above_limit'])
    y_train = train_data['income_above_limit']

    X_train = X_train[feature_importance_df['Feature'].tolist()]

    print(f"Step 4: Loading and preprocessing test data for {filename}...")
    # Step 5: Load and preprocess the test data
    test_data_file = os.path.join(sample_dir, filename.replace("_train.csv", "_test.csv"))
    test_data = pd.read_csv(test_data_file)
    X_test = test_data.drop(['income_above_limit', 'kfold'], axis=1)
    y_true = test_data['income_above_limit']
    X_test = X_test[feature_importance_df['Feature'].tolist()]

    print(f"Step 5: Creating an ExtraTreesClassifier with hyperparameters...")
    # Step 6: Create an ExtraTreesClassifier with the specified hyperparameters
    model = lgb.LGBMClassifier(**hyperparameters)

    print(f"Step 6: Initializing the Min-Max scaler...")
    # Step 7: Initialize the Min-Max scaler
    #scaler = MinMaxScaler()

    print(f"Step 7: Fitting and transforming the scaler on training data...")
    # Step 8: Fit and transform the scaler on your training data
    #X_train = scaler.fit_transform(X_train)

    print(f"Step 8: Transforming test data using the same scaler...")
    # Step 9: Transform the test data using the same scaler
    #X_test = scaler.transform(X_test)

    print(f"Step 9: Initializing a label encoder...")
    # Step 10: Initialize a label encoder
    label_encoder = LabelEncoder()

    print(f"Step 10: Fitting the label encoder on unique class labels and transforming target variable...")
    # Step 11: Fit the label encoder on the unique class labels and transform the target variable
    y_train = label_encoder.fit_transform(y_train)
    y_true = label_encoder.transform(y_true)

    print(f"Step 11: Training the model...")
    # Step 12: Train the model
    model.fit(X_train, y_train)

    # Step 13: Save the trained model to a file
    model_file_path = os.path.join(model_dir, f"{filename.replace('.csv', '_model.joblib')}")
    dump(model, model_file_path)

    print(f"Step 12: Making predictions on test data...")
    # Step 13: Make predictions on the test data
    y_pred = model.predict(X_test)

    print(f"Step 13: Calculating the F1 score for {filename}...")
    # Step 14: Calculate the F1 score
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    return f1_weighted

# Step 15: Load data from the YAML file
with open("data/final/top_f1_score.yaml", "r") as yaml_file:
    data = yaml.safe_load(yaml_file)

# Step 16: Extract the list of file names
file_names = data['top_f1_score']

# Step 17: Define the directory to save feature importances
feature_importance_dir = "data/final/feature_importances"
sample_dir = "data/processed/sample_data"

# Step 18: Define the directory to save trained models
model_dir = "model"

# Step 19: Load your DataFrame (replace with your actual file path)
df = pd.read_csv('data/final/feature_importances/top_model_results.csv')

# Step 20: Initialize lists to store model filenames and F1 scores
model_filenames = []
f1_scores = []

# Step 21: Loop through each train file (feature importance file) and find the top model
for index, row in df.iterrows():
    # Step 22: Get the filename and hyperparameters
    filename = row["Filename"]
    hyperparameters = eval(row["Best Model Hyperparameters"])

    # Step 23: Train and evaluate the model, then append the filename and F1 score to the lists
    f1 = train_and_evaluate_model(filename, hyperparameters)
    model_filenames.append(filename)
    f1_scores.append(f1)

# Step 24: Create a DataFrame for plotting
plot_data = pd.DataFrame({'Model Filename': model_filenames, 'F1 Weighted Score': f1_scores})

# Step 25: Plot the F1 scores using Plotly Express
fig = px.bar(plot_data, x='Model Filename', y='F1 Weighted Score', color='Model Filename')
fig.update_xaxes(categoryorder='total descending')  # Order by descending F1 score
fig.update_layout(title='F1 Weighted Scores for Models', xaxis_title='Model Filename', yaxis_title='F1 Weighted Score')
fig.show()