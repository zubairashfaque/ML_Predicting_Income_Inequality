import streamlit as st
import pandas as pd
import plotly.express as px
from data import get_data
import io
import pandas as pd
import streamlit as st

path_csv = "data/raw/data.csv"
df = pd.read_csv(path_csv)

df = df.head(1000)
def df_info(df):
    df.columns = df.columns.str.replace(' ', '_')
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()

    df_info = s.split('\n')

    counts = []
    names = []
    nn_count = []
    dtype = []
    for i in range(5, len(df_info) - 3):
        line = df_info[i].split()
        counts.append(line[0])
        names.append(line[1])
        nn_count.append(line[2])
        dtype.append(line[4])

    df_info_dataframe = pd.DataFrame(
        data={'#': counts, 'Column': names, 'Non-Null Count': nn_count, 'Data Type': dtype})
    return df_info_dataframe.drop('#', axis=1)


def df_isnull(df):
    res = pd.DataFrame(df.isnull().sum()).reset_index()
    res['Percentage'] = round(res[0] / df.shape[0] * 100, 2)
    res['Percentage'] = res['Percentage'].astype(str) + '%'
    return res.rename(columns={'index': 'Column', 0: 'Number of null values'})


def number_of_outliers(df):
    df = df.select_dtypes(exclude='object')

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    ans = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    df = pd.DataFrame(ans).reset_index().rename(columns={'index': 'column', 0: 'count_of_outliers'})
    return df


def space(num_lines=1):
    for _ in range(num_lines):
        st.write("")


def sidebar_space(num_lines=1):
    for _ in range(num_lines):
        st.sidebar.write("")


def sidebar_multiselect_container(massage, arr, key):
    container = st.sidebar.container()
    select_all_button = st.sidebar.checkbox("Select all for " + key + " plots")
    if select_all_button:
        selected_num_cols = container.multiselect(massage, arr, default=list(arr))
    else:
        selected_num_cols = container.multiselect(massage, arr, default=arr[0])

    return selected_num_cols


def main():
    st.title("Data Visualization Dashboard")

    # Load your data
    df = get_data()

    all_visuals = ['Info', 'NA Info', 'Descriptive Analysis', 'Target Analysis',
                   'Distribution of Numerical Columns', 'Count Plots of Categorical Columns',
                   'Box Plots', 'Outlier Analysis', 'Variance of Target with Categorical Columns']

    # Visualization selection at the center
    st.subheader("Choose which visualizations you want to see 👇")
    vizuals = st.multiselect("Select visualizations", all_visuals)

    if 'Info' in vizuals:
        st.subheader('Info:')
        c1, c2, c3 = st.columns([1, 2, 1])
        c2.dataframe(df_info(df))

    if 'NA Info' in vizuals:
        st.subheader('NA Value Information:')
        if df.isnull().sum().sum() == 0:
            st.write('There is not any NA value in your dataset.')
        else:
            c1, c2, c3 = st.columns([0.5, 2, 0.5])
            c2.dataframe(df_isnull(df), width=1500)
            space(2)

    if 'Descriptive Analysis' in vizuals:
        st.subheader('Descriptive Analysis:')
        st.dataframe(df.describe())

    if 'Target Analysis' in vizuals:
        st.subheader("Select target column:")
        target_column = st.selectbox("", df.columns, index=len(df.columns) - 1)

        st.subheader("Histogram of target column")
        fig = px.histogram(df, x=target_column)
        c1, c2, c3 = st.columns([0.5, 2, 0.5])
        c2.plotly_chart(fig)

    num_columns = df.select_dtypes(exclude='object').columns
    cat_columns = df.select_dtypes(include='object').columns

    if 'Distribution of Numerical Columns' in vizuals:

        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = sidebar_multiselect_container('Choose columns for Distribution plots:', num_columns,
                                                              'Distribution')
            st.subheader('Distribution of numerical columns')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_num_cols)):
                        break

                    fig = px.histogram(df, x=selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width=True)
                    i += 1

    if 'Count Plots of Categorical Columns' in vizuals:

        if len(cat_columns) == 0:
            st.write('There is no categorical columns in the data.')
        else:
            selected_cat_cols = sidebar_multiselect_container('Choose columns for Count plots:', cat_columns, 'Count')
            st.subheader('Count plots of categorical columns')
            i = 0
            while (i < len(selected_cat_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_cat_cols)):
                        break

                    fig = px.histogram(df, x=selected_cat_cols[i], color_discrete_sequence=['indianred'])
                    j.plotly_chart(fig)
                    i += 1

    if 'Box Plots' in vizuals:
        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = sidebar_multiselect_container('Choose columns for Box plots:', num_columns, 'Box')
            st.subheader('Box plots')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_num_cols)):
                        break

                    fig = px.box(df, y=selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width=True)
                    i += 1

    if 'Outlier Analysis' in vizuals:
        st.subheader('Outlier Analysis')
        c1, c2, c3 = st.columns([1, 2, 1])
        c2.dataframe(number_of_outliers(df))

    if 'Variance of Target with Categorical Columns' in vizuals:

        df_1 = df.dropna()

        high_cardi_columns = []
        normal_cardi_columns = []

        for i in cat_columns:
            if (df[i].nunique() > df.shape[0] / 10):
                high_cardi_columns.append(i)
            else:
                normal_cardi_columns.append(i)

        if len(normal_cardi_columns) == 0:
            st.write('There is no categorical columns with normal cardinality in the data.')
        else:

            st.subheader('Variance of target variable with categorical columns')
            model_type = st.radio('Select Problem Type:', ('Regression', 'Classification'), key='model_type')
            selected_cat_cols = sidebar_multiselect_container('Choose columns for Category Colored plots:',
                                                              normal_cardi_columns, 'Category')

            if 'Target Analysis' not in vizuals:
                target_column = st.selectbox("Select target column:", df.columns, index=len(df.columns) - 1)

            i = 0
            while (i < len(selected_cat_cols)):

                if model_type == 'Regression':
                    fig = px.box(df_1, y=target_column, color=selected_cat_cols[i])
                else:
                    fig = px.histogram(df_1, color=selected_cat_cols[i], x=target_column)

                st.plotly_chart(fig, use_container_width=True)
                i += 1

            if high_cardi_columns:
                if len(high_cardi_columns) == 1:
                    st.subheader('The following column has high cardinality, that is why its boxplot was not plotted:')
                else:
                    st.subheader(
                        'The following columns have high cardinality, that is why its boxplot was not plotted:')
                for i in high_cardi_columns:
                    st.write(i)

                st.write('<p style="font-size:140%">Do you want to plot anyway?</p>', unsafe_allow_html=True)
                answer = st.selectbox("", ('No', 'Yes'))

                if answer == 'Yes':
                    for i in high_cardi_columns:
                        fig = px.box(df_1, y=target_column, color=i)
                        st.plotly_chart(fig, use_container_width=True)
if __name__ == "__main__":
    main()