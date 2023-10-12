import pandas as pd
import lightgbm as lgb
from collections import OrderedDict
from sklearn.metrics import f1_score
import logging
import pickle
import bz2
import os
from skopt import BayesSearchCV

# Configure logging
logging.basicConfig(filename='Project_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the file paths
test_file = 'data/processed/sample_data/oversampler_smote_0_test.csv'
train_file = 'data/processed/sample_data/oversampler_smote_0_train.csv'
selected_features_file = 'data/selected_features_test.csv'

try:
    logging.info("Training Started")
    # Load the training data
    train_data = pd.read_csv(train_file)

    # Select the specified features
    selected_features = ['education', 'age', 'total_employed', 'occupation_code', 'industry_code', 'gender', 'gains',
                         'industry_code_main', 'tax_status', 'stocks_status']

    X_train = train_data[selected_features]
    y_train = train_data['income_above_limit']

    # Define the Extra Tree model with hyperparameters
    hyperparameters = OrderedDict()

    # Define the parameter search space for Bayesian optimization
    param_space = {
        'n_estimators': (50, 200),  # Number of boosting rounds
        'learning_rate': (0.01, 1.0),  # Learning rate
        'max_depth': (3, 15),  # Maximum tree depth
        'num_leaves': (2, 50),  # Maximum number of leaves in a tree
        'min_child_samples': (1, 20),  # Minimum number of samples required to create a leaf
        'subsample': (0.5, 1.0),  # Fraction of samples used for training trees
        'colsample_bytree': (0.5, 1.0),  # Fraction of features used for training trees
    }

    # Create a LightGBM classifier
    model = lgb.LGBMClassifier()

    # Use BayesSearchCV for hyperparameter tuning
    search = BayesSearchCV(model, param_space, n_iter=30, cv=5, scoring='f1_weighted', n_jobs=-1)

    # Fit the search to find the best hyperparameters
    search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = search.best_params_
    logging.info("Best Hyperparameters: {}".format(best_params))
    print("Best Hyperparameters:", best_params)

    # Train the model with the best hyperparameters
    best_model = lgb.LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # Load the test data
    test_data = pd.read_csv(test_file)

    # Select the same features for the test data
    X_test = test_data[selected_features]
    y_test = test_data['income_above_limit']

    # Predict using the trained model
    y_pred = best_model.predict(X_test)

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
    model_pickle_file = 'model/extra_trees_model.pkl'
    with open(model_pickle_file, 'wb') as f_out:
        pickle.dump(best_model, f_out)

    logging.info("Model saved using pickle to: {}".format(model_pickle_file))
    print("Model saved using pickle to:", model_pickle_file)

    # Save the trained model using bzip2 compression
    compressed_model_file = 'model/extra_trees_model.pkl.bz2'
    with bz2.BZ2File(compressed_model_file, 'wb') as f_out:
        pickle.dump(best_model, f_out)

    logging.info("Compressed model saved to: {}".format(compressed_model_file))
    print("Compressed model saved to:", compressed_model_file)
    # Remove the pickle file after it's been successfully compressed
    os.remove(model_pickle_file)
    logging.info("Pickle model file removed: {}".format(model_pickle_file))
    print("Pickle model file removed:", model_pickle_file)
    logging.info("Training Ended")
except Exception as e:
    logging.error("An error occurred: {}".format(str(e)))
