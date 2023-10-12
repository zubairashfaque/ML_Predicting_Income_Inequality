# Import necessary libraries
import pandas as pd
import lightgbm as lgb
from collections import OrderedDict
from sklearn.metrics import f1_score
import logging
import pickle
import bz2
import os

# Configure logging
logging.basicConfig(filename='Project_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the file paths
test_file = 'data/processed/sample_data/oversampler_smote_0_test.csv'
train_file = 'data/processed/sample_data/oversampler_smote_0_train.csv'
selected_features_file = 'data/selected_features_test.csv'
model_file = 'model/LGBMClassifier_model.pkl'

try:
    logging.info(
        "-----------------------------------------Training Started-----------------------------------------")
    # Load the training data
    train_data = pd.read_csv(train_file)

    # Select the specified features
    selected_features = ['education', 'age', 'total_employed', 'occupation_code', 'industry_code', 'gender', 'gains',
                            'industry_code_main', 'tax_status', 'stocks_status']

    X_train = train_data[selected_features]
    y_train = train_data['income_above_limit']

    # Define the Extra Tree model with hyperparameters
    best_hyperparameters = {
        'colsample_bytree': 0.5329019931356901,
        'learning_rate': 0.20317186261723266,
        'max_depth': 6,
        'min_child_samples': 10,
        'n_estimators': 174,
        'num_leaves': 48,
        'subsample': 1
    }
    print(X_train.columns)

    model = lgb.LGBMClassifier(**best_hyperparameters)

    # Train the model
    model.fit(X_train, y_train)

    # Load the test data
    test_data = pd.read_csv(test_file)

    # Select the same features for the test data
    X_test = test_data.drop(['income_above_limit', 'kfold'], axis=1)
    X_test = test_data[selected_features]
    y_test = test_data['income_above_limit']

    # Predict using the trained model
    y_pred = model.predict(X_test)

    # Calculate F1-weighted score
    f1_weighted_score = f1_score(y_test, y_pred, average='weighted')

    # Print the F1-weighted score
    logging.info("F1 Weighted Score: {}".format(f1_weighted_score))
    print("F1 Weighted Score:", f1_weighted_score)

    # Save the selected features to a CSV file
    X_test.to_csv(selected_features_file, index=False)
    logging.info("Selected features saved to: {}".format(selected_features_file))
    print("Selected features saved to:", selected_features_file)

    # Save the trained model using pickle
    model_pickle_file = 'model/LGBMClassifier_model.pkl'
    with open(model_pickle_file, 'wb') as f_out:
        pickle.dump(model, f_out)

    logging.info("Model saved using pickle to: {}".format(model_pickle_file))
    print("Model saved using pickle to:", model_pickle_file)

    logging.info(
        "-----------------------------------------Training Ended-----------------------------------------")
except Exception as e:
    logging.error("An error occurred: {}".format(str(e)))
