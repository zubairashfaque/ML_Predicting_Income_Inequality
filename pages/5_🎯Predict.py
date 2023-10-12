import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve  # Import roc_curve here
import lightgbm as lgb
import plotly.express as px
import plotly.figure_factory as ff


# Define Streamlit app with a large centered heading including a dynamic element
st.markdown("<h1 style='text-align: center; color: #008080; font-size: 58px;'>🔮 Welcome to the Income Prediction App! 🔮</h1>", unsafe_allow_html=True)


# Load the pre-trained model
model = joblib.load("model/LGBMClassifier_model.pkl")

# Load the LabelEncoders for categorical features
ordinal_encoders = joblib.load("data/processed/ordinal_encoder.pkl")
test_file = 'data/processed/sample_data/oversampler_smote_0_test.csv'

# Function to make predictions
def predict_income(input_data):

    if input_data.isnull().any().any():
        return ["Missing values in input data"]
    # Perform label encoding for categorical features
    categorical_features = [
            'education',
            'gender',
            'tax_status',
            'industry_code_main'
        ]
    # Handle NaN values by filling them with appropriate values
    for feature in categorical_features:
        input_data[feature] = input_data[feature].fillna("unknown")

    # Apply ordinal encoding to the selected columns
    input_data[categorical_features] = ordinal_encoders.transform(input_data[categorical_features])

    # Preprocess the input data to match the model's expectations
    # You need to preprocess the data based on how the model was trained

    # Make predictions
    prediction = model.predict(input_data)

    return prediction

selected_features = ['education', 'age', 'total_employed', 'occupation_code', 'industry_code', 'gender', 'gains',
                            'industry_code_main', 'tax_status', 'stocks_status']

# Load the test data
test_data = pd.read_csv(test_file)

# Select the same features for the test data
X_test = test_data.drop(['income_above_limit', 'kfold'], axis=1)
X_test = test_data[selected_features]
y_test = test_data['income_above_limit']
# Calculate model accuracy
y_pred = model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate AUC-ROC
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

# Streamlit app
st.title("🎯 Income Prediction App 🎯")

# Center the input fields for the user
st.markdown('<h2 style="text-align: center;"> User Input </h2>', unsafe_allow_html=True)

# Create input fields for user
col1, col2 = st.columns(2)

# Define feature values
industry_code_values = [
    0, 41, 33, 35, 22, 29, 38, 42, 43, 4, 8, 50, 5, 23, 2, 37, 34, 45, 3, 31,
    36, 1, 47, 39, 19, 32, 25, 6, 30, 15, 16, 24, 40, 48, 27, 44, 9, 12, 11, 21,
    17, 28, 7, 13, 14, 18, 49, 20, 26, 46, 51, 10
]

occupation_code_values = [
    0, 26, 31, 2, 17, 36, 33, 8, 10, 40, 29, 39, 23, 34, 43, 38, 3, 35, 19, 25,
    27, 24, 30, 1, 32, 15, 5, 28, 37, 12, 4, 16, 7, 14, 44, 18, 13, 9, 42, 21, 41,
    22, 11, 45, 6, 20, 46
]

gains_values = [
    0, 991, 2964, 7688, 99999, 1848, 7298, 4508, 15024, 2290, 114, 3674,
    14344, 20051, 6849, 1173, 2414, 10605, 5178, 1409, 25124, 11678, 594, 4650,
    7896, 3411, 2062, 5060, 3908, 401, 15020, 4787, 4386, 3103, 2329, 1506,
    914, 6514, 6767, 18481, 2228, 1086, 1455, 14084, 10520, 3471, 4101, 4064,
    2635, 2829, 8614, 15831, 2597, 27828, 2407, 10566, 7443, 2354, 5013, 4934,
    1831, 3325, 2993, 3418, 3942, 2907, 2346, 7262, 4416, 2176, 2885, 1055,
    5721, 1424, 3273, 1797, 9386, 3464, 2580, 2174, 2009, 3137, 6097, 6418,
    3432, 13550, 4865, 22040, 2936, 2050, 6497, 2977, 5455, 9562, 3887, 2105,
    1151, 4931, 2538, 7978, 2202, 1264, 2653, 7430, 3818, 3781, 3456, 2774,
    6612, 5556, 4687, 2463, 6723, 1111, 2961, 25236, 6360, 2387, 1471, 34095,
    3800, 4594, 41310, 2227, 1140, 2601, 2036, 1090, 8530, 1731, 1639, 2098
]

industry_code_main_values = [
    'Not in universe or children', 'Hospital services', 'Retail trade',
    'Finance insurance and real estate', 'Manufacturing-nondurable goods',
    'Transportation', 'Business and repair services',
    'Medical except hospital', 'Education', 'Construction',
    'Manufacturing-durable goods', 'Public administration', 'Agriculture',
    'Other professional services', 'Mining',
    'Utilities and sanitary services', 'Private household services',
    'Personal services except private HH', 'Wholesale trade',
    'Communications', 'Entertainment', 'Social services',
    'Forestry and fisheries', 'Armed Forces'
]

education_values = [
    'High school graduate', '12th grade no diploma', 'Children',
    'Bachelors degree(BA AB BS)', '7th and 8th grade', '11th grade',
    '9th grade', 'Masters degree(MA MS MEng MEd MSW MBA)', '10th grade',
    'Associates degree-academic program', '1st 2nd 3rd or 4th grade',
    'Some college but no degree', 'Less than 1st grade',
    'Associates degree-occup /vocational',
    'Prof school degree (MD DDS DVM LLB JD)', '5th or 6th grade',
    'Doctorate degree(PhD EdD)'
]

tax_status_values = [
    'Head of household', 'Single', 'Nonfiler', 'Joint both 65+',
    'Joint both under 65', 'Joint one under 65 & one 65+'
]

gender_values = ["Male", "Female"]

stocks_status_values = ["Own", "None", "Other"]

with col1:
    education = st.selectbox("Education", education_values)
    age = st.number_input("Age", min_value=18, max_value=100)
    total_employed = st.number_input("Total Employed", min_value=0, max_value=5)
    occupation_code = st.selectbox("Occupation Code", occupation_code_values)
    industry_code = st.selectbox("Industry Code", industry_code_values)
    gender = st.selectbox("Gender", gender_values)

# Modify the stocks_status input field
with col2:
    gains = st.selectbox("Gains", gains_values)
    industry_code_main = st.selectbox("Industry Code Main", industry_code_main_values)
    tax_status = st.selectbox("Tax Status", tax_status_values)
    stocks_status = st.number_input("Stocks Status", min_value=0, max_value=6000, value=0, step=1)


# Define an income limit for prediction
income_limit = 50000

# Create a DataFrame from user inputs
user_data = pd.DataFrame({
    "education": [education],
    "age": [age],
    "total_employed": [total_employed],
    "occupation_code": [occupation_code],
    "industry_code": [industry_code],
    "gender": [gender],
    "gains": [gains],
    "industry_code_main": [industry_code_main],
    "tax_status": [tax_status],
    "stocks_status": [stocks_status]
})

# Add a button for prediction
if st.button("Predict Income"):
    # Make predictions
    prediction = predict_income(user_data)

    # Display the prediction
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.markdown("<h2 style='color: red;'>🔴 The predicted income is below the limit</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: green;'>🟢 The predicted income is above the limit</h2>", unsafe_allow_html=True)

    # Display model name and accuracy
    st.subheader("Model Information")
    st.write(f"Model Name: LightGBM Classifier")
    st.write(f"Model Accuracy: {model_accuracy:.2%}")


    # Create an interactive confusion matrix using Plotly
    # Create an interactive confusion matrix using Plotly
    st.subheader("Confusion Matrix")

    # Increase the size of the Plotly figure
    fig = ff.create_annotated_heatmap(z=cm, x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'],
                                      colorscale='Blues')
    fig.update_layout(title='Confusion Matrix', autosize=False, width=600, height=400)
    st.plotly_chart(fig)

    # Define the background color (light gray)
    background_color = "lightgray"

    # Display an interactive AUC-ROC Curve using Plotly with a light gray background
    st.subheader("AUC-ROC Curve")
    fig = px.line(x=fpr, y=tpr, labels={"x": "False Positive Rate", "y": "True Positive Rate"}, title="AUC-ROC Curve")
    fig.add_shape(
        type="line", line=dict(dash="dash"),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(
        plot_bgcolor=background_color,
    )
    st.plotly_chart(fig)

    # Display a tree diagram (change 0 to the desired tree number)
    st.subheader("Tree Diagram")
    graph_data = lgb.create_tree_digraph(model, tree_index=0, show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
    st.graphviz_chart(graph_data)