import os
import yaml
import pandas as pd
import lightgbm as lgb
from skopt import BayesSearchCV
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
# Define the parameter search space
param_space = {
    'n_estimators': (50, 200),           # Number of boosting rounds
    'learning_rate': (0.01, 1.0),       # Learning rate
    'max_depth': (3, 15),               # Maximum tree depth
    'num_leaves': (2, 50),              # Maximum number of leaves in a tree
    'min_child_samples': (1, 20),       # Minimum number of samples required to create a leaf
    'subsample': (0.5, 1.0),            # Fraction of samples used for training trees
    'colsample_bytree': (0.5, 1.0),     # Fraction of features used for training trees
}

# Create a BayesSearchCV object for ExtraTreesClassifier hyperparameter tuning
lgb_classifier  = lgb.LGBMClassifier()
# Create a BayesSearchCV object
bayes_search = BayesSearchCV(
    lgb_classifier,
    param_space,
    n_iter=50,  # Number of iterations (adjust as needed)
    scoring='f1_weighted',  # Scoring metric for optimization
    cv=5,  # Number of cross-validation folds
    n_jobs=-1,  # Use all available CPU cores
    verbose=1,  # Enable verbose output
    refit=True,  # Refit the best model on the entire dataset
    random_state=42,  # Set a random seed for reproducibility
)

# Load data from the YAML file
with open("data/final/top_f1_score.yaml", "r") as yaml_file:
    data = yaml.safe_load(yaml_file)

# Extract the list of file names
file_names = data['top_f1_score']

# Directory to save feature importances
feature_importance_dir = "data/final/feature_importances"

# Initialize an empty list to store top models and their results
top_model_results = []

# Loop through each train file (feature importance file) and find the top model
for train_file in file_names:
    print(f"Processing file: {train_file}")
    new_filename = train_file.replace("_train.csv", "_Extra Trees_top_10_features.csv")
    file_path = os.path.join(feature_importance_dir, new_filename)
    feature_importance_df = pd.read_csv(file_path)
    print("Top 10 features:", feature_importance_df['Feature'].tolist())

    # Load the corresponding train data file
    train_data_file = os.path.join('data/processed/sample_data', train_file)
    train_data = pd.read_csv(train_data_file)

    # Split the data into features (X) and target (y)
    X_train = train_data.drop(columns=['income_above_limit'])
    y_train = train_data['income_above_limit']

    # Select only the top 10 features based on feature importance
    selected_features = feature_importance_df['Feature'].tolist()
    X_train = X_train[selected_features]

    # Perform Bayesian Optimization on the training data
    bayes_search.fit(X_train, y_train)

    # Get the best hyperparameters and the best score
    best_params = bayes_search.best_params_
    best_score = bayes_search.best_score_

    # Train an ExtraTreesClassifier with the best hyperparameters
    lgb_classifier.set_params(**best_params)
    lgb_classifier.fit(X_train, y_train)

    # Define the filename for saving the model
    model_filename = train_file.replace(".csv", "_model.joblib")

    # Save the trained LightGBM model to the specified filename
    model_save_path = os.path.join("model", model_filename)
    joblib.dump(lgb_classifier, model_save_path)
    print("LightGBM model saved as:", model_save_path)

    # Append the top model and its results to the list
    top_model_results.append({
        'Filename': train_file,
        'Model_name': 'LGBMClassifier',
        'Best Model Hyperparameters': best_params,
        'Best Model Score (F1 Weighted)': best_score
    })

# Create a DataFrame from the top model results list
top_model_results_df = pd.DataFrame(top_model_results)

# Save the top model results DataFrame to a CSV file
top_model_results_file_path = os.path.join(feature_importance_dir, 'top_model_results.csv')
top_model_results_df.to_csv(top_model_results_file_path, index=False)

# Print a message indicating the script has completed execution
print("Script execution completed.")