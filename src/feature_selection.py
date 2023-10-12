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
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.pipeline import Pipeline
from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    SmartCorrelatedSelection,
)
import numpy as np
import warnings
import logging

# Configure logging
logging.basicConfig(filename='Project_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress FutureWarning about is_categorical_dtype
warnings.filterwarnings('ignore', category=FutureWarning, module='xgboost')

# Load data from the YAML file
try:
    with open("data/final/top_f1_score.yaml", "r") as yaml_file:
        data = yaml.safe_load(yaml_file)

    # Extract the list of file names
    file_names = data['top_f1_score']

    # List to store combined feature files
    combined_feature_files = []

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
        'LightGBM': lgb.LGBMClassifier(verbose=-1),  # Suppress LightGBM verbose output
        'CatBoost': cb.CatBoostClassifier(silent=True),  # Suppress CatBoost verbose output
        'HistGradientBoosting': HistGradientBoostingClassifier()
    }

    # Initialize F1 scores dictionary
    f1_scores = {}

    # Directory to save feature importances
    feature_importance_dir = "data/final/feature_importances"
    os.makedirs(feature_importance_dir, exist_ok=True)

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
            try:
                model.fit(X_train, y_train)  # Train the model on X_train

                # Get feature importances if available
                if hasattr(model, 'feature_importances_'):
                    feature_importances = model.feature_importances_
                elif hasattr(model, 'estimators_'):
                    # For ensemble methods like AdaBoost, you can aggregate feature importances from base estimators
                    feature_importances = np.mean([estimator.feature_importances_ for estimator in model.estimators_], axis=0)
                else:
                    # Handle cases where the model doesn't provide feature importances
                    feature_importances = None

                if feature_importances is not None:
                    # Create a DataFrame with feature names and their importances
                    feature_importance_df = pd.DataFrame({'Feature': X_train_t.columns, 'Importance': feature_importances})

                    # Sort the DataFrame by importance (descending order)
                    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

                    # Save the top 10 features based on importance
                    top_10_features = feature_importance_df['Feature'][:10]
                    top_10_features_file = os.path.join(feature_importance_dir,
                                                        f'{train_file.replace("_train.csv", "")}_{model_name}_top_10_features.csv')
                    top_10_features.to_csv(top_10_features_file, index=False)
            except Exception as e:
                logging.error(f"Error for {train_file} and {model_name}: {str(e)}")

    # Reset the warning filters after your XGBoost-related code
    warnings.resetwarnings()
    # Print a message indicating the script has completed execution
    logging.info("Script execution completed.")

except Exception as e:
    logging.error(f"Error: {str(e)}")
