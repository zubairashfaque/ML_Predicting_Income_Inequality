import os
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import shap
import matplotlib.pyplot as plt
from joblib import load, dump
# Mathematical functions
import math
from IPython.display import display

# Step 1: Load data from the YAML file
with open("data/final/top_f1_score.yaml", "r") as yaml_file:
    data = yaml.safe_load(yaml_file)

# Step 2: Extract the list of file names (assuming one model and one test file)
file_names = data['top_f1_score']  # Select the first model and test file

# Step 3: Define the directory for test data
test_data_dir = "data/processed/sample_data"

# Directory to save feature importances
feature_importance_dir = "data/final/feature_importances"

# Step 4: Initialize a list to store SHAP values
shap_values_list = []


test_data_file = os.path.join(test_data_dir, "oversampler_smote_0_test.csv")
test_data = pd.read_csv(test_data_file, nrows=100)
#test_data = pd.read_csv(test_data_file)
X_test = test_data.drop(columns=['income_above_limit', 'kfold'])
y_true = test_data['income_above_limit']
model_file = "oversampler_smote_0_test.csv"
new_filename = model_file.replace("_test.csv", "_LightGBM_top_10_features.csv")
file_path = os.path.join(feature_importance_dir, new_filename)
feature_importance_df = pd.read_csv(file_path)
selected_features = feature_importance_df['Feature'].tolist()
X_test = X_test[selected_features]
# Step 9: Apply necessary preprocessing (MinMax scaling and Label Encoding)
scaler = MinMaxScaler()
X_test = scaler.fit_transform(X_test)
# Loop through each model file and generate SHAP values (for one model)
label_encoder = LabelEncoder()
y_true = label_encoder.fit_transform(y_true)
print(X_test)

# Convert X_test back to a DataFrame with feature names as columns
X_test = pd.DataFrame(X_test, columns=selected_features)

# Now, X_test_df is a DataFrame with feature names as columns
print(X_test)
model_path = os.path.join("model", "oversampler_smote_0_train_model.joblib")
loaded_model = load(model_path)

y_pred = loaded_model.predict(X_test)



# get shap values
explainer = shap.Explainer(loaded_model)
shap_values = explainer(X_test)

print(np.shape(shap_values))

# Find the class with the highest predicted probability for each observation
predicted_class_indices = np.argmax(loaded_model.predict_proba(X_test), axis=1)
print(predicted_class_indices)

# Create a waterfall plot for the first observation's SHAP values

shap.plots.waterfall(shap_values[0, :, predicted_class_indices[0]])

# calculate mean SHAP values for each class
mean_0 = np.mean(np.abs(shap_values.values[:, :, 0]), axis=0)
mean_1 = np.mean(np.abs(shap_values.values[:, :, 1]), axis=0)



new_shap_values = []
for i, pred in enumerate(y_pred):
    # get shap values for predicted class
    new_shap_values.append(shap_values.values[i][:, pred])

# replace shap values
shap_values.values = np.array(new_shap_values)
print(shap_values.shape)


shap.plots.bar(shap_values)

shap.plots.beeswarm(shap_values)


# Loading JavaScript library
shap.initjs()

# Sampling from test data predictors
X_test_sample = X_test.sample(100)

# Predicted values corresponding to the sample
y_pred_extree_sample = np.array(pd.Series(data = y_pred, index = X_test.index)[X_test.index])

# Explainer
explainer = shap.TreeExplainer(loaded_model)

# Computing SHAP values based on the sample
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values = shap_values, features = X_test_sample, plot_type = 'bar')


# Reverse encoder for the target variable
target_reverse_encoder = {'Above limit': 0, 'Below limit': 1}


# Force plot
shap.initjs()
row = math.floor(len(y_pred_extree_sample)/2)
for i in range(np.array(shap_values).shape[0]):
    print(target_reverse_encoder[i+1])
    display(shap.force_plot(base_value = explainer.expected_value[i],
                            shap_values = shap_values[i][row],
                            features = X_test_sample.values[row],
                            feature_names = X_test_sample.columns))
print("Prediction: {}".format(target_reverse_encoder[y_pred_extree_sample[row]]))


# Waterfall plot
for i in range(np.array(shap_values).shape[0]):
    print(target_reverse_encoder[i+1])
    display(shap.waterfall_plot(shap.Explanation(values = shap_values[i][row],
                                                 base_values = explainer.expected_value[i],
                                                 data = X_test_sample.iloc[row],
                                                 feature_names = X_test_sample.columns.tolist())))
    print(" ")
print("Prediction: {}".format(target_reverse_encoder[y_pred_extree_sample[row]]))


# Decision plot
for i in range(np.array(shap_values).shape[0]):
    print(target_reverse_encoder[i+1])
    display(shap.decision_plot(base_value = explainer.expected_value[i],
                               shap_values = shap_values[i][row],
                               features = X_test_sample.iloc[row, :],
                               feature_names = X_test_sample.columns.tolist()))
    print(" ")
print("Prediction: {}".format(target_reverse_encoder[y_pred_extree_sample[row]]))