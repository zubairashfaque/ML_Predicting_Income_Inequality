import streamlit as st
st.set_page_config(layout = "wide", page_icon = 'logo.png', page_title='EDA')
# Add custom CSS to widen the content within a single column
st.write(
    f"""
    <style>
    .reportview-container .main .block-container{{
        max-width: 90%;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Page title and introduction
st.title("Income Inequality Prediction")
st.write("Explore income inequality prediction with machine learning.")

# Description section
st.header("Description 📋")
description_text = """
Income inequality, where income is distributed unevenly among a population, is a pressing issue in many developing nations worldwide. The rapid advancement of AI and automation has the potential to exacerbate this problem. This solution aims to reduce the cost and improve the accuracy of monitoring key population indicators, such as income levels, between census years. This information can empower policymakers to better manage and mitigate income inequality on a global scale.
"""
st.markdown(description_text)

# Problem Statement section
st.header("Problem Statement 📊")
problem_statement_text = """
The target feature is **income_above_limit**, a binary-class variable. The challenge is to create a machine learning model that predicts whether an individual earns above or below a certain income threshold. Your model's performance will be evaluated using the **F1-score** metric.
"""
st.markdown(problem_statement_text)

# Additional resources and links
st.header("Additional Resources and Links 🌐")