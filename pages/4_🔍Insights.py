import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency



# Define Streamlit app with a large centered heading including 🔍 symbols
st.markdown("<h1 style='text-align: center; color: #008080; font-size: 58px;'>🔍 Insights 🔍</h1>", unsafe_allow_html=True)

# Rest of your Streamlit code


# Load your data
path_csv = "data/raw/data.csv"
df = pd.read_csv(path_csv)

# Define the category of interest (Above limit)
category_of_interest = 'Above limit'

# Function to calculate conditional probabilities
def calculate_conditional_probabilities(df):
    conditional_probabilities = {}
    for education_category in df['education'].unique():
        # Filter the DataFrame for the specified education category and income_above_limit category of interest
        filtered_df = df[(df['education'] == education_category) & (df['income_above_limit'] == category_of_interest)]

        # Calculate the conditional probability
        conditional_probability = len(filtered_df) / len(df[df['income_above_limit'] == category_of_interest])

        # Store the conditional probability in a dictionary
        conditional_probabilities[education_category] = conditional_probability
    return conditional_probabilities

# Create education category mapping
education_mapping = {
    ' High school graduate': 'upto 12th',
    ' 12th grade no diploma': 'upto 12th',
    ' Children': 'under 12th',
    ' Bachelors degree(BA AB BS)': 'Higher Education',
    ' 7th and 8th grade': 'under 12th',
    ' 11th grade': 'under 12th',
    ' 9th grade': 'under 12th',
    ' Masters degree(MA MS MEng MEd MSW MBA)': 'Higher Education',
    ' 10th grade': 'under 12th',
    ' Associates degree-academic program': 'Higher Education',
    ' 1st 2nd 3rd or 4th grade': 'under 12th',
    ' Some college but no degree': 'upto 12th',
    ' Less than 1st grade': 'under 12th',
    ' Associates degree-occup /vocational': 'Higher Education',
    ' Prof school degree (MD DDS DVM LLB JD)': 'Higher Education',
    ' 5th or 6th grade': 'under 12th',
    ' Doctorate degree(PhD EdD)': 'Higher Education'
}

# Apply the mapping to create a new column 'education_category'
df['education_category'] = df['education'].map(education_mapping)

# Define Streamlit app
st.title("Education vs. Income Analysis")

# Interactive Question 1: Does education have any impact on income?
question_1 = st.checkbox("Question 1: Does education have any impact on income?")

if question_1:
    st.markdown("### Analysis of Education and Income")
    st.markdown("Let's explore the impact of education on income.")

    # Calculate conditional probabilities
    conditional_probabilities = calculate_conditional_probabilities(df)

    # Find the category with the highest conditional probability
    max_probability_category = max(conditional_probabilities, key=conditional_probabilities.get)

    # Find the category with the lowest conditional probability
    min_probability_category = min(conditional_probabilities, key=conditional_probabilities.get)

    # Create a DataFrame for visualization
    plot_df = pd.DataFrame({
        'Education Categories': conditional_probabilities.keys(),
        'Conditional Probabilities': conditional_probabilities.values()
    })

    # Find the index of the highest and lowest probabilities
    max_index = plot_df['Conditional Probabilities'].idxmax()
    min_index = plot_df['Conditional Probabilities'].idxmin()

    # Sort the DataFrame by probability
    plot_df = plot_df.sort_values(by='Conditional Probabilities', ascending=False)

    # Create a beautiful color palette
    color_palette = px.colors.qualitative.Plotly

    # Create a bar plot using Plotly
    fig = px.bar(
        plot_df,
        x='Education Categories',
        y='Conditional Probabilities',
        color_discrete_sequence=color_palette,
        labels={'Conditional Probabilities': 'Conditional Probability'},
        title="Conditional Probabilities of Education Categories given 'Above limit' Income"
    )

    # Highlight the bars for the highest and lowest probabilities
    color_list = [color_palette[0]] * len(plot_df)
    color_list[max_index] = color_palette[1]
    color_list[min_index] = color_palette[2]

    # Identify most common and least common categories
    most_common = plot_df.iloc[0]['Education Categories']
    least_common = plot_df.iloc[-1]['Education Categories']

    # Add annotations for most common and least common categories
    fig.add_annotation(x=most_common, y=plot_df.iloc[0]['Conditional Probabilities'] + 0.02,
                       text=f'High prob.: {most_common}', showarrow=True,
                       arrowhead=2, arrowcolor=color_palette[2], arrowwidth=2)
    fig.add_annotation(x=least_common, y=plot_df.iloc[-1]['Conditional Probabilities'] + 0.02,
                       text=f'Lowset prob.: {least_common}', showarrow=True,
                       arrowhead=2, arrowcolor=color_palette[1], arrowwidth=2)

    # Update the title, axis labels, and axis tick font size
    fig.update_layout(
        title_text="Conditional Probabilities of Education Categories given 'Above limit' Income",
        title_x=0.5,  # Centered title
        xaxis_title='Education Categories',
        yaxis_title='Conditional Probability',
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12))
    )

    # Update y-axis to make it even longer
    fig.update_yaxes(range=[0, max(plot_df['Conditional Probabilities']) + 0.2])

    # Show the plot with highlighted bars
    st.plotly_chart(fig)

    # Additional Analysis
    st.subheader("Additional Analysis")

    # Calculate and display percentages for each category
    st.markdown("### Income Analysis by Education Category")
    st.markdown("The following analysis provides insights into income distribution based on education categories.")

    for category in education_mapping.values():
        category_df = df[df['education_category'] == category]
        below_limit_percentage = (category_df['income_above_limit'] == 'Below limit').mean() * 100
        above_limit_percentage = (category_df['income_above_limit'] == 'Above limit').mean() * 100

        st.write(f"For individuals in the education category '{category}':")
        st.write(f"- Percentage with earnings below the income limit: {below_limit_percentage:.2f}%")
        st.write(f"- Percentage with earnings above the income limit: {above_limit_percentage:.2f}%")
        st.write("")

    st.markdown("### 💡 **Observations:**")
    st.markdown(
        "The education category with the highest conditional probability is 'Bachelors degree(BA AB BS)' with a probability of 31.96%, indicating that individuals with this level of education are more likely to have earnings above the income limit.")
    st.markdown(
        "On the other hand, the education category 'Children' has the lowest conditional probability of 0.00%, suggesting that individuals in this category are very unlikely to have earnings above the income limit.")

    st.markdown("#### Income Analysis by Education Category:")
    st.markdown(
        "For individuals in the education category 'upto 12th,' 95.31% of them have earnings below the income limit, while only 4.69% have earnings above the income limit.")
    st.markdown(
        "In the 'under 12th' education category, a significant 99.64% of individuals have earnings below the income limit, with only 0.36% having earnings above the limit.")
    st.markdown(
        "In the 'Higher Education' category, 78.54% of individuals have earnings below the income limit, while 21.46% have earnings above the limit.")

    st.markdown(
        "These insights highlight the strong influence of education on income levels. Notably, individuals with higher education, such as a bachelor's degree, are more likely to earn above the income limit. However, the data also suggests that a considerable number of individuals with education up to the 12th grade have earnings below the limit, emphasizing the need for further exploration and potentially addressing income disparities in this group.")

# Define Streamlit app
st.title("Citizenship vs. Income Analysis")

# Interactive Question 2: Does citizenship have any impact on income? (Notice the unique key)
question_2 = st.checkbox("Question 2: Does citizenship have any impact on income?", key="question_2")

if question_2:
    st.markdown("### Analysis of Citizenship and Income")
    st.markdown("Let's explore the impact of Citizenship on income. First, we will take a look at the cross-tabulation table and the chi-squared test for independence.")
    # Create a cross-tabulation table
    cross_table = pd.crosstab(df['income_above_limit'], df['citizenship'])

    # Create a heatmap to visualize the cross-tabulation table
    fig = go.Figure(data=go.Heatmap(
        z=cross_table.values,
        x=cross_table.columns,
        y=cross_table.index,
        colorscale='Viridis',  # You can choose a different color scale
    ))

    # Customize the heatmap appearance
    fig.update_layout(
        title='Cross-Tabulation of Income vs Citizenship',
        xaxis_title='Citizenship',
        yaxis_title='Income',
    )

    # Show the heatmap
    st.plotly_chart(fig)

    # Perform the chi-squared test for independence
    chi2, p, dof, expected = chi2_contingency(cross_table)

    # Print the test results
    st.write("Chi-Squared Statistic:", chi2)
    st.write("P-Value:", p)
    st.write("Degrees of Freedom:", dof)
    st.write("Expected Frequencies Table:")
    st.dataframe(pd.DataFrame(expected, columns=cross_table.columns, index=cross_table.index))

    # Check if the p-value is less than the chosen significance level (e.g., 0.05)
    alpha = 0.05
    if p < alpha:
        st.write("Reject the null hypothesis: There is a significant relationship between income_above_limit and citizenship.")
    else:
        st.write("Fail to reject the null hypothesis: There is no significant relationship between income_above_limit and citizenship.")

    # Calculate and display percentages for Native and Non-Native citizenship
    native_below_limit_percentage = (len(df[(df['citizenship'] == 'Native') & (df['income_above_limit'] == 'Below limit')]) / len(df[df['citizenship'] == 'Native'])) * 100
    non_native_below_limit_percentage = (len(df[(df['citizenship'] != 'Native') & (df['income_above_limit'] == 'Below limit')]) / len(df[df['citizenship'] != 'Native'])) * 100

    st.subheader("Income Analysis by Citizenship:")
    st.write(f"Percentage of Native citizenship with income below limit: {native_below_limit_percentage:.2f}%")
    st.write(f"Percentage of Non-Native citizenship with income below limit: {non_native_below_limit_percentage:.2f}%")

    st.markdown("💡 **Observations:**")
    st.markdown("We performed the chi-squared test for independence, which resulted in 'Reject the null hypothesis: There is a significant relationship between income_above_limit and citizenship.' This means that there is evidence to suggest a statistically significant association or relationship between the two categorical variables, 'income_above_limit' and 'citizenship.'")
    st.markdown("In the cross-tabulation table, we can observe the counts of individuals categorized by their citizenship status and income level. The table provides insight into the distribution of individuals among different citizenship categories and their respective income levels.")
    st.markdown("Specifically, when looking at the 'Native' citizenship category:")
    st.markdown("- The majority of individuals have income 'Above limit,' with 11,710 individuals falling into this category.")
    st.markdown("- A smaller number of individuals, 36, have income 'Below limit.'")
    st.markdown("In contrast, when examining citizenship categories other than 'Native':")
    st.markdown("- A significant proportion of individuals have income 'Above limit,' indicating a relatively higher number of individuals with income above the limit in these categories.")
    st.markdown("- However, a slightly higher percentage of individuals in these categories also have income 'Below limit.'")
    st.markdown("It appears that there is little to no discrimination, as both individuals with 'Native' citizenship and those with 'Non-Native' citizenship exhibit very similar rates of income below the limit.")

# Define Streamlit app
st.title("Gender vs. Income Analysis")

# Interactive Question 3: Income inequality in gender? (Notice the unique key)
question_3 = st.checkbox("Question 3: Income inequality in gender?", key="question_3")

if question_3:
    st.markdown("### Analysis of Income Inequality in Gender")
    st.markdown("Let's explore income inequality based on gender.")

    # Convert 'income_above_limit' to numerical values
    df['income_numeric'] = df['income_above_limit'].apply(lambda x: 0 if x == 'Below limit' else 1)

    # Separate data for males and females
    male_data = df[df['gender'] == ' Male']
    female_data = df[df['gender'] == ' Female']

    # Perform t-test
    from scipy import stats

    t_stat, p_value = stats.ttest_ind(male_data['income_numeric'], female_data['income_numeric'])

    alpha = 0.05  # Set your desired significance level

    if p_value < alpha:
        st.write("Reject the null hypothesis: There is a significant difference in income between males and females.")
    else:
        st.write(
            "Fail to reject the null hypothesis: There is no significant difference in income between males and females.")

    # Visualization: Create visualizations to compare income distributions for males and females
    st.markdown("#### Visualization: Income Distribution by Gender")

    import plotly.express as px

    # Calculate the proportions of income levels by gender
    male_income_proportions = male_data['income_above_limit'].value_counts(normalize=True)
    female_income_proportions = female_data['income_above_limit'].value_counts(normalize=True)

    # Create a DataFrame for visualization
    plot_df = pd.DataFrame({
        'Gender': ['Male', 'Female'],
        'Income Above Limit': [male_income_proportions['Above limit'], female_income_proportions['Above limit']],
        'Income Below Limit': [male_income_proportions['Below limit'], female_income_proportions['Below limit']]
    })

    # Create an interactive horizontal bar chart using Plotly
    fig = px.bar(
        plot_df,
        y='Gender',
        x=['Income Above Limit', 'Income Below Limit'],
        color_discrete_sequence=['blue', 'pink'],
        labels={'value': 'Proportion', 'variable': 'Income Level'},
        title='Income Distribution by Gender',
        orientation='h'  # Horizontal orientation
    )

    # Customize the layout
    fig.update_layout(
        xaxis_title='Proportion',
        yaxis_title='Gender',
        legend_title='Income Level',
        showlegend=True,
        barmode='group'
    )

    # Show the plot
    st.plotly_chart(fig)

    # Descriptive Statistics: Calculate summary statistics for each group to get an overview
    st.markdown("#### Descriptive Statistics: Summary Statistics by Gender")

    import tabulate

    # Calculate the summary statistics for income distribution by gender
    male_summary = male_data['income_above_limit'].describe()
    female_summary = female_data['income_above_limit'].describe()

    # Create a dictionary to store the summary statistics
    summary_dict = {
        'Gender': ['Male', 'Female'],
        'Count': [male_summary['count'], female_summary['count']],
        'Unique': [male_summary['unique'], female_summary['unique']],
        'Top': [male_summary['top'], female_summary['top']],
        'Frequency': [male_summary['freq'], female_summary['freq']]
    }

    # Format the summary statistics as a table
    table = tabulate.tabulate(summary_dict, headers='keys', tablefmt='pretty')

    # Display the table
    st.text(table)

    # Calculate the percentage of males and females with income below the limit
    males_below_limit = len(df[(df['gender'] == ' Male') & (df['income_above_limit'] == 'Below limit')])
    females_below_limit = len(df[(df['gender'] != ' Male') & (df['income_above_limit'] == 'Below limit')])

    total_males = len(df[df['gender'] == ' Male'])
    total_females = len(df[df['gender'] != ' Male'])

    percentage_males_below_limit = (males_below_limit / total_males) * 100
    percentage_females_below_limit = (females_below_limit / total_females) * 100

    st.markdown(f'Males with income below limit: {percentage_males_below_limit:.2f}%')
    st.markdown(f'Females with income below limit: {percentage_females_below_limit:.2f}%')

    st.markdown("💡 **Observations:**")
    st.markdown(
        "We conducted a t-test to compare the income levels between males and females. The test resulted in 'Reject the null hypothesis: There is a significant difference in income between males and females.' This indicates that there is statistical evidence suggesting a significant income difference based on gender.")
    st.markdown(
        "Further analysis revealed that out of the total 100,715 males in the dataset, approximately 89.86% have income below the limit. In contrast, among the 108,784 females, about 97.44% fall below the income limit. While there is indeed a statistically significant difference in income levels between males and females, it's important to note that the practical difference is not very high.")
    st.markdown(
        "In summary, there is evidence of a gender-related income disparity, but the magnitude of this difference is relatively low.")