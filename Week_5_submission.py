import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# Set page configuration to wide mode
st.set_page_config(layout="wide")

# Inject custom CSS to further reduce side margins
st.markdown(
    """
    <style>
        .block-container {
            padding-left: 5rem !important;
            padding-right: 5rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and problem statement
st.title("Loan Approval Prediction 📊")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Dataset Overview", "Features", "Datatypes", "Sunburst", "Correlation Matrix", "Prediction"])

with tab1:
    st.subheader("Problem statement")
    st.markdown("""
                The approval of loans is a critical decision for financial institutions, influenced by a multitude of factors ranging from applicant demographics to financial behavior.
                It is also important from an applicant's perspective as to how their personal financial behaviors, housing stability, and credit history influence approval odds, empowering them to make informed decisions when seeking loans.
                To optimize risk assessment and lending strategies, it is essential to identify the key drivers of loan approval outcomes.
                This project aims to analyze a dataset containing applicant profiles and loan characteristics and to predict the likelihood of loan approval based on the attributes available in the dataset.
                """)

    # View dataset
    st.markdown("##### Dataset")

    @st.cache_data
    def load_data():
        df = pd.read_csv('loan_dataset.csv')
        return df

    df = load_data()

    # # df = pd.read_csv('loan_dataset.csv')
    # new_column_names = {
    #                     'person_age': 'age',
    #                     'person_income': 'income',
    #                     'person_home_ownership': 'home_ownership',
    #                     'person_emp_length': 'emp_length',
    #                     'loan_amnt': 'loan_amount',
    #                     'cb_person_default_on_file': 'cb_default',
    #                     'cb_person_cred_hist_length': 'cb_cred_hist',
    #                     }
    # df = df.rename(columns=new_column_names)

    # # Drop outliers
    # df = df[df['age'] != 123]
    # df = df[df['emp_length'] != 123]
    # df = df[df['income'] <= 1000000]

    

    # Dataset info
    st.markdown(f" The dataset is taken from [kaggle](https://www.kaggle.com/competitions/playground-series-s4e10/data). There are no missing values. After removing a few outliers, it has {len(df)} rows and 13 columns.")

    # View dataset
    st.dataframe(df)

    # View dataset columns
    st.markdown("##### Dataset columns")
    cols = {
                'Columns': ['id',
                            'age',
                            'income',
                            'home_ownership',
                            'emp_length',
                            'loan_intent',
                            'loan_grade',
                            'loan_amount',
                            'loan_int_rate',
                            'loan_percent_income',
                            'cb_default',
                            'cb_cred_hist',
                            'loan_status'],
            
            'Description': ['Unique identifier for each loan',
                            'Age of loan applicant',
                            'Annual income of applicant',
                            'Home ownership status of the applicant (RENT, MORTGAGE, OWN, OTHER)',
                            'Length of employment of loan applicant (0 indicates less than 1 year of employment)',
                            'Purpose of taking loan (EDUCATION, MEDICAL, PERSONAL, VENTURE, DEBTCONSOLIDATION, HOMEIMPROVEMENT)',
                            'Credit grade assigned to the loan that reflects risk level [A (low risk) to G (high risk)]',
                            'Loan amount requested',
                            'Annual interest rate charged on the loan (as a percentage)',
                            'Loan amount as a percentage of the applicant’s annual income',
                            'Indicates if the applicant has a history of defaulting on loans (Y or N)',
                            'Length of the applicant’s credit history (in years)',
                            'Target variable indicating whether the loan was approved (1) or denied (0)']
            }

    cols_df = pd.DataFrame(cols)
    st.table(cols_df)

    with tab2:
        st.markdown("##### Column Inspection")
        col1, col2 = st.columns(2)

        with col1:
            column_type = st.selectbox('Select datatype of column', ['Categorical', 'Numerical'], key="selectbox_1", help="Select the type of column you want to visualize")
            if column_type == 'Numerical':
                selected_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
            else:
                selected_cols = [col for col in df.columns if df[col].dtype == 'object']
        with col2:
            selected_column = st.selectbox("Select a column to visualize", selected_cols, help="Select the column you want to visualize")
                

        def plot_graphs(df, column):
            # Define color mapping
            color_discrete_map = {0: 'lightcoral', 1: 'mediumaquamarine'}

            if column:
                if pd.api.types.is_numeric_dtype(df[column]):
                    
                    # Histogram
                    fig_hist = px.histogram(df, x=column, title="Histogram", nbins=30, marginal="box", opacity=0.7, color_discrete_sequence=['skyblue'])

                    # Box plot
                    fig_box = px.box(df, y=column, title="Box Plot", color_discrete_sequence=['skyblue'])

                    # KDE plot
                    fig_kde = go.Figure()
                    for loan_status in df['loan_status'].unique():
                        subset = df[df['loan_status'] == loan_status]
                        fig_kde.add_trace(go.Histogram(
                            x=subset[column],
                            histnorm='probability density',
                            opacity=0.6,
                            marker=dict(color=color_discrete_map[loan_status]),
                            name=f'loan_status {loan_status}'
                        ))

                    fig_kde.update_layout(
                        title="Kernel density estimate (KDE) plot",
                        xaxis_title=column,
                        yaxis_title="Density",
                        barmode='overlay'
                    )
                    fig_kde.update_traces(opacity=0.6)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_hist, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    st.plotly_chart(fig_kde, use_container_width=True)

                else:
                    value_counts = df[column].value_counts().reset_index()
                    value_counts.columns = [column, 'count']

                    # Bar plot
                    fig_bar = px.bar(value_counts, x=column, y='count',
                                    title="Bar plot", labels={column: column, 'count': 'Count'}, color_discrete_sequence=['skyblue'])

                    # Pie chart
                    # Create a pie chart
                    fig_pie = px.pie(
                        value_counts,
                        names=column,
                        values='count',
                        title="Pie Chart",
                        color_discrete_sequence=px.colors.qualitative.Pastel  # Custom color palette
                    )

                    # Update pie chart hover template
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}</br>'
                    )

                    # Grouped bar plot
                    fig_grouped_bar = go.Figure()

                    for status, color in color_discrete_map.items():
                        filtered_df = df[df['loan_status'] == status]
                        fig_grouped_bar.add_trace(
                            go.Bar(
                                x=filtered_df[column].value_counts().index,
                                y=filtered_df[column].value_counts().values,
                                name=f'{'loan_status'}={status}',
                                marker_color=color
                            )
                        )

                    fig_grouped_bar.update_layout(
                        title_text='Count plot',
                        title_font_size=16,
                        xaxis_title=column,
                        yaxis_title='Count',
                        barmode='group',  # Grouped bars
                        legend_title='loan_status',
                        showlegend=True
                    )

                    fig_grouped_bar.update_xaxes(tickangle=45)

                    # Stacked percentage bar plot
                    crosstab = pd.crosstab(df[column], df['loan_status'], normalize='index') * 100

                    fig_stacked_bar = go.Figure()

                    for i, status in enumerate(crosstab.columns):
                        fig_stacked_bar.add_trace(
                            go.Bar(
                                x=crosstab.index,
                                y=crosstab[status],
                                name=f'{'loan_status'}={status}',
                                marker_color=color_discrete_map[i],
                                textposition='auto'
                            )
                        )

                    fig_stacked_bar.update_layout(
                        title_text='Stacked percentage bar plot',
                        title_font_size=16,
                        xaxis_title=column,
                        yaxis_title=f'Percentage of {'loan_status'}',
                        barmode='stack',  # Stacked bars
                        legend_title='loan_status',
                        showlegend=True
                    )

                    fig_stacked_bar.update_xaxes(tickangle=45)

                    col1, col2 = st.columns(2)
                    col3, col4 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_bar, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col3:
                        st.plotly_chart(fig_grouped_bar, use_container_width=True)
                    with col4:
                        st.plotly_chart(fig_stacked_bar, use_container_width=True)

        plot_graphs(df, selected_column)


with tab3:
    st.markdown("##### Cumulative Analysis")
    col1, col2 = st.columns(2)

    with col1:
        col_type = st.selectbox('Select datatype of column', ['Categorical', 'Numerical'], key="selectbox_2", help="Select the type of column you want to visualize")

    with col2:
        if col_type == 'Numerical':
            plot_type = st.selectbox('Select type of plot', ['Histogram', 'Boxplot', 'KDE'], help="Select the type of plot you want to visualize")
        else:
            plot_type = st.selectbox('Select type of plot', ['Bar', 'Grouped Bar', 'Stacked Bar'], help="Select the type of plot you want to visualize")

    # Function to create subplots layout
    def create_subplots(num_plots, num_cols=4):
        num_rows = (num_plots + num_cols - 1) // num_cols
        return num_rows, num_cols

    # Generic function to plot columns in a grid layout
    def plot_columns(df, columns, num_rows, num_cols, plot_function, **kwargs):
        """
        Generic function to plot columns in a grid layout.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
            columns (list): List of columns to plot.
            num_rows (int): Number of rows in the grid.
            num_cols (int): Number of columns in the grid.
            plot_function (function): The function to generate the plot for a single column.
            **kwargs: Additional arguments to pass to the plot_function.
        """
        for i in range(num_rows):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                idx = i * num_cols + j
                if idx < len(columns):
                    column = columns[idx]
                    fig = plot_function(df, column, **kwargs)
                    cols[j].plotly_chart(fig, use_container_width=True)

    # Plot-specific functions
    def plot_histogram(df, column, **kwargs):
        return px.histogram(df, x=column, nbins=30, title=f'{column}', labels={column: column})

    def plot_boxplot(df, column, **kwargs):
        return px.box(df, y=column, title=f'{column}', labels={column: 'Values'})

    def plot_kde(df, column, **kwargs):
        fig = go.Figure()
        for loan_status in df['loan_status'].unique():
            subset = df[df['loan_status'] == loan_status]
            fig.add_trace(go.Histogram(
                x=subset[column],
                histnorm='probability density',
                opacity=0.6,
                marker=dict(color=kwargs['color_discrete_map'][loan_status]),
                name=f'loan_status {loan_status}'
            ))
        fig.update_layout(
            title=f'{column}',
            xaxis_title=column,
            yaxis_title="Density",
            barmode='overlay',
            showlegend=False
        )
        fig.update_traces(opacity=0.6)
        return fig

    def plot_bar(df, column, **kwargs):
        value_counts = df[column].value_counts().reset_index()
        value_counts.columns = [column, 'count']
        return px.bar(value_counts, x=column, y='count', title=f'{column}', labels={column: column})

    def plot_grouped_bar(df, column, **kwargs):
        fig = go.Figure()
        for status, color in kwargs['color_discrete_map'].items():
            filtered_df = df[df['loan_status'] == status]
            fig.add_trace(
                go.Bar(
                    x=filtered_df[column].value_counts().index,
                    y=filtered_df[column].value_counts().values,
                    name=f'loan_status={status}',
                    marker_color=color
                )
            )
        fig.update_layout(
            title_text=f'{column}',
            title_font_size=16,
            xaxis_title=column,
            yaxis_title='Count',
            barmode='group',
            # legend_title='loan_status',
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)
        return fig

    def plot_stacked_bar(df, column, **kwargs):
        crosstab = pd.crosstab(df[column], df['loan_status'], normalize='index') * 100
        fig = go.Figure()
        for k, status in enumerate(crosstab.columns):
            fig.add_trace(
                go.Bar(
                    x=crosstab.index,
                    y=crosstab[status],
                    name=f'loan_status={status}',
                    marker_color=kwargs['color_discrete_map'][k],
                    textposition='auto'
                )
            )
        fig.update_layout(
            title_text=f'{column}',
            title_font_size=16,
            xaxis_title=column,
            yaxis_title=f'Percentage of loan_status',
            barmode='stack',
            # legend_title='loan_status',
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)
        return fig

    if col_type == 'Numerical':
        numerical_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'id']
        num_plots = len(numerical_columns)
        num_rows, num_cols = create_subplots(num_plots)

        if plot_type == 'Histogram':
            st.markdown('**Histograms of numerical columns**')
            plot_columns(df, numerical_columns, num_rows, num_cols, plot_histogram)
            with st.expander("Findings"):
                st.markdown("""
                                - Dataset predominantly consists of younger population. It is reasonable to assume that the employement length and credit history almost follows a similar distribution due to this reason.
                                - Majority of the income lies within 200k, with very few making over 800k.
                                - Loan amount and interest rate seem to be random
                                - Loan percent is mostly under 40% with much of the data predominantly under 20%
                                - Loan status distribution of 0 vs 1 is almost 5:1""")

        elif plot_type == 'Boxplot':
            st.markdown('**Box plot of numerical columns**')
            plot_columns(df, numerical_columns, num_rows, num_cols, plot_boxplot)
            with st.expander("Findings"):
                st.markdown("""
                                - Much of our learning here mirrors similar finidings from the histograms above. However we can visualize the outliers much more clearly.""")

        elif plot_type == 'KDE':
            st.markdown('**KDE plots of numerical columns (by loan_status)**')
            color_discrete_map = {0: 'lightcoral', 1: 'mediumaquamarine'}
            plot_columns(df, numerical_columns, num_rows, num_cols, plot_kde, color_discrete_map=color_discrete_map)
            with st.expander("Findings"):
                st.markdown("""
                                - Loan status does not seem to vary with age and credit history length
                                - Lower ranges of income and employement length seems to have higher peaks for loan_status = 1
                                - Higher loan amount, interest rate and loan percent of income seems to have higher peaks for loan_status = 1""")

    elif col_type == 'Categorical':
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object' and col != 'id']
        num_plots = len(categorical_columns)
        num_rows, num_cols = create_subplots(num_plots)

        if plot_type == 'Bar':
            st.markdown('**Bar plots of categorical columns**')
            plot_columns(df, categorical_columns, num_rows, num_cols, plot_bar)
            with st.expander("Findings"):
                st.markdown("""
                                - Home ownership: Rent and Mortgage are most common modes of applicants' housing
                                - Loan intent: Loans are taken in large number across almost all categories with EDUCATION being the highest and HOMEINPROVEMENT being the lowest among them.
                                - Loan grade: Low risk loans (A, B, C) are more in number than high risk loans (D and above)
                                - Default on file: Applicants who don't have a record of default (N) outweigh defaulters (Y) by almost 5 to 1. """)

        elif plot_type == 'Grouped Bar':
            st.markdown('**Grouped bar plots of categorical columns (by loan_status)**')
            color_discrete_map = {0: 'lightcoral', 1: 'mediumaquamarine'}
            plot_columns(df, categorical_columns, num_rows, num_cols, plot_grouped_bar, color_discrete_map=color_discrete_map)
            with st.expander("Findings"):
                st.markdown("""
                                - loan_status = 0 seems to dominate in almost every set of categorical variables except for high risk loans (D to G) under loan_grade.""")

        elif plot_type == 'Stacked Bar':
            st.markdown('**Stacked bar plots of categorical columns (by loan_status)**')
            color_discrete_map = {0: 'lightcoral', 1: 'mediumaquamarine'}
            plot_columns(df, categorical_columns, num_rows, num_cols, plot_stacked_bar, color_discrete_map=color_discrete_map)
            with st.expander("Findings"):
                st.markdown("""
                                - These stacked bar charts effectively visualizes the percentage distribution of target variable among the categorical variables.""")

with tab4:
    st.markdown("##### Sunburst charts")
    st.write("These charts helps us visualize the distribution of categorical variables among one another")

    categorical_columns = [col for col in df.columns if df[col].dtype == 'object' and col != 'id']
    df_cat = df[categorical_columns]

    # Allow user to select columns
    parent_column = st.selectbox("Select Parent Column", df_cat.columns, help="Primary category upon which the chart will be based")
    child_column = st.selectbox("Select Child Column", [col for col in categorical_columns if col != parent_column], help="Secondary category that will be nested within the parent category")

    # Group the data by the parent and child columns and count occurrences
    grouped_df = (
        df_cat.groupby([parent_column, child_column])
        .size()
        .reset_index(name='count')
    )

    # Calculate the total count
    total_count = grouped_df['count'].sum()

    # Add a new column for percentages
    grouped_df['percentage'] = (grouped_df['count'] / total_count) * 100

    # Create the sunburst chart using percentages
    fig = px.sunburst(
        grouped_df,
        path=[parent_column, child_column],  # Hierarchical structure
        values='percentage',                # Use percentages for segment sizes
        title="Sunburst Chart"
    )

    # Customize hover text to show only percentages
    fig.update_traces(
        hovertemplate=(
            "<b>%{label}</b><br>" +
            "Percentage: %{value:.2f}%<extra></extra>"
        )
    )

    # Increase the figure size
    fig.update_layout(
        width=650,   # Width of the figure in pixels
        height=650   # Height of the figure in pixels
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)



with tab5:
    st.write("##### Correlation Matrix Tables (after label encoding `cb_default`)")

    df_1 = df.copy()
    df_1['cb_default'] = df_1['cb_default'].map({'N': 0, 'Y': 1})

    label_encoder = LabelEncoder()
    df_1['loan_grade'] = label_encoder.fit_transform(df_1['loan_grade'])

    numerical_columns_1 = [col for col in df_1.columns if df_1[col].dtype in ['int64', 'float64'] and col != 'id']
    df_1_numerical = df_1[numerical_columns_1]
    corr_matrix_1 = df_1_numerical.corr()

    st.table(corr_matrix_1.style.background_gradient(cmap="coolwarm").format("{:.2f}"))

    with st.expander("Findings"):
        st.markdown("""
                    - **Inferences:**
                        - From this correlation plot, we can verify that higher the loan grade, higher the interest rate. But since their correlation is as high as 0.94, we can proceed to drop loan_grade and keep only loan_interest_rate to avoid multicollinearity.
                        - Age and credit history having a high correlation of 0.88. It is obvious that higher the age, the longer their credit histories would be. We can drop the person_age columnm as well.

                    - **Other observations:**
                        - cb_default_on_file has a correlation of 0.55 with loan_grade and 0.94 with loan_interest_rate. This means that the loans for defaulters are usually associated with high risk. Hence, the higher interest rate and grade.
                        - loan_amount has a high correlation of 0.65 with loan_percent_income
                        - loan_amount has a positive correlation of 0.34 with person_income

                    - **Positive correlations with target variable (loan_status):**
                        - loan_grade
                        - loan_interest_rate
                        - loan_percent_income
                        - cb_person_default_on_file
                    """)

    df_1.drop(['age', 'loan_grade'], axis=1, inplace=True)

    # st.markdown('---')
    st.text("")
    st.write("##### Correlation Matrix Tables (after one-hot encoding `loan_intent` and `home_ownership`)")
    df_2 = df_1.copy()

    # One-Hot Encoding
    one_hot_encoder = OneHotEncoder(sparse_output=False)  # Keep all categories
    columns_to_encode = ['loan_intent', 'home_ownership']
    encoded_features = one_hot_encoder.fit_transform(df_2[columns_to_encode])

    # Create a DataFrame for the encoded features
    encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(columns_to_encode), index=df_2.index)

    # Drop the original columns and join the encoded columns
    df_encoded = df_2.drop(columns=columns_to_encode).join(encoded_df)

    # Select numerical columns
    numerical_columns_2 = [col for col in df_encoded.columns if df_encoded[col].dtype in ['int64', 'float64'] and col != 'id']
    df_encoded_numerical = df_encoded[numerical_columns_2]

    # Compute the correlation matrix
    corr_matrix_2 = df_encoded_numerical.corr()

    # Visualize the correlation matrix using a heatmap
    plt.figure(figsize=(15, 12))  # Set the figure size
    sns.heatmap(corr_matrix_2, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Matrix Heatmap")
    st.pyplot(plt)

    # Close the plot to avoid memory issues
    plt.close()

    with st.expander("Findings"):
        st.markdown("""
                    - **Key inferences:**
                        - loan_amount vs loan_int_rate (0.65): indicates that higher loan amounts tend to come with higher interest rates.
                        - cb_default_on_file vs loan_int_rate (0.5): indicates that people who defaulted on a loan in the past tend to receive higher interest rates as they are considered high risk.
                        - loan_percent_income vs loan_interest (0.38): suggests that as the percentage of income dedicated to the loan increases, the interest rate also tends to increase.
                        - person_income vs loan_amount (0.34): suggests that higher-income individuals tend to take larger loans.
                        - home_ownership vs person_income and person_emp_length:
                            - Mortgage ownership shows a positive correlation with income (0.30) and employement_length (0.29), indicating that mortgage payment is most common among people with higher incomes and those who are well into their career years.
                            - Renting is negatively correlated with income (-0.28) and employement_length (-0.29), indicating renters generally have lower incomes and is usually associated with people who are in their early stages of career.
                        - employment_length vs income (0.18): this weak to moderate positive correlation suggests that longer employment is somewhat associated with higher income.

                    - **Correlations with target_variable (loan_status):**
                        - loan_intents have weak correlations with loan_status, meaning the purpose of the loan might not be a strong predictor.
                        - person_income has a weak to moderate negative correlated with loan_status (-0.18)
                        - loan_amount (0.14) and person_default_on_file (0.19) has weak to moderate positive correlations
                        - loan_interest_rate (0.34) and loan_percent_income (0.38) have moderate positive correlations
                        - home_ownership_MORTGAGE (-0.20) and home_ownership_RENT (0.24) have moderate correlations
                    """)


with tab6:
    data = df_encoded.copy()
    st.write("##### Dataframe for modeling:")
    st.write(data.head())
    st.write("---")

    # Initialize session state for metrics_df
    if 'metrics_df' not in st.session_state:
        st.session_state.metrics_df = pd.DataFrame(columns=['Model/Metric', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC'])

    def prepare_data(data):
        # Define features and target
        X = data.drop(columns=['id', 'loan_status'])
        y = data['loan_status']

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test

    def evaluate_model(y_test, y_pred, y_pred_proba, model_name):
        # Evaluate the model
        accuracy = round(accuracy_score(y_test, y_pred), 2)
        precision = round(precision_score(y_test, y_pred, average='binary'), 2)
        recall = round(recall_score(y_test, y_pred, average='binary'), 2)
        f1 = round(f1_score(y_test, y_pred, average='binary'), 2)
        auc = round(roc_auc_score(y_test, y_pred_proba), 2)
        
        # Add metrics to the DataFrame
        new_row = pd.DataFrame({
            'Model/Metric': [model_name],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1-score': [f1],
            'AUC': [auc]
        })
        
        global metrics_df
        # Ensure new_row does not contain empty or all-NA columns
        new_row = new_row.dropna(axis=1, how='all')
        # Ensure metrics_df does not contain empty or all-NA columns before concatenation
        st.session_state.metrics_df.dropna(axis=1, how='all', inplace=True)
        # Append the new row to the DataFrame
        st.session_state.metrics_df = pd.concat([st.session_state.metrics_df, new_row], ignore_index=True)

    st.write("##### Input a model and its parameters")
    col1, col2 = st.columns(2)
    with col1:
        model_names = ['Logistic Regression', 'KNN', 'ADABoost', 'XGBoost', 'Random Forest']
        selected_model = st.selectbox(
            "Choose a Model",  # Label for the select box
            model_names,       # List of options
            index=0            # Default selected option (index 0 is 'Logistic Regression')
        )
    with col2:
        thershold_value = st.number_input('Choose a threshold to color cells', min_value=0.50, max_value=1.0, value=0.75, step=0.05, help="Values great than this will be colored green and less than this will be colored red")

    # Logistic Regression
    if selected_model == 'Logistic Regression':
        X_train, X_test, y_train, y_test = prepare_data(data)

        model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000, solver='liblinear')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        evaluate_model(y_test, y_pred, y_pred_proba, 'Logistic Regression')


    # KNN
    elif selected_model == 'KNN':
        col1, col2, col3, col4 = st.columns(4)

        neighbours = col1.number_input('n_neighbors', min_value=1, max_value=20, value=5, step=1, help="Number of neighbors to use")
        weights = col2.radio('weights', ['uniform', 'distance'], index=0, horizontal=True, help="Weight function used in prediction")
        algo = col3.radio('algorithm', ["auto", "ball_tree", "kd_tree", "brute"], index=0, horizontal=True, help="Algorithm used to compute the nearest neighbors")

        X_train, X_test, y_train, y_test = prepare_data(data)

        model = KNeighborsClassifier(
            n_neighbors=5,      # Number of neighbors to use
            weights='uniform',  # Use uniform weights
            algorithm='auto',   # Choose algorithm used to compute nearest neighbors
            n_jobs=-1           # Use all available processors
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        evaluate_model(y_test, y_pred, y_pred_proba, f'KNN ({neighbours} neighbours, {weights} weight, {algo} algo)')

    # ADA
    elif selected_model == 'ADABoost':

        col1, col2 = st.columns(2)

        estimators = col1.number_input('n_estimators', min_value=50, max_value=1000, value=100, step=50, help="Number of weak learners to train iteratively (in increments of 50)")
        rate = col2.number_input('learning_rate', min_value=0.01, max_value=0.5, value=0.1, step=0.01, help="Weight applied to each weak learner (in increments of 0.01)")

        X_train, X_test, y_train, y_test = prepare_data(data)

        # Apply SMOTE to the training data
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        base_estimator = DecisionTreeClassifier(max_depth=1, class_weight='balanced', random_state=42)

        model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=estimators,
            algorithm='SAMME',
            learning_rate=rate,
            random_state=42
        )

        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        evaluate_model(y_test, y_pred, y_pred_proba, f'ADABoost ({estimators} est, {rate} rate)')

    # XG
    elif selected_model == 'XGBoost':
        col1, col2, col3, col4 = st.columns(4)

        estimators = col1.number_input('n_estimators', min_value=50, max_value=1000, value=100, step=50, help="Number of boosting rounds (trees)")
        depth = col2.number_input('max_depth', min_value=3, max_value=10, value=5, step=1, help="Maximum depth of each decision tree")
        rate = col3.number_input('learning_rate', min_value=0.01, max_value=0.5, value=0.1, step=0.01, help="Step size shrinkage applied to tree contributions.")
        weight = col4.number_input('Enter x (class_weights = {0\: 1, 1: neg/pos * x})', min_value=1.0, max_value=5.0, value=1.0, step=0.5, help="Multiplication factor for the minority class")

        # XGBoost with different weights
        X_train, X_test, y_train, y_test = prepare_data(data)
        neg, pos = y_train.value_counts() # Calculate class imbalance ratio
        scale_pos_weight = neg / pos   # Increase weight for the minority class
        adjusted_weight = scale_pos_weight * weight

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            # use_label_encoder=False,
            scale_pos_weight=adjusted_weight,  # Balances classes by giving more weight to the minority class
            max_depth=depth,
            learning_rate=rate,
            n_estimators=estimators,
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        evaluate_model(y_test, y_pred, y_pred_proba, f"XGBoost ({estimators} est, {depth} dep, {rate} rate, neg/pos * {weight})")


    # RF
    else:
        col1, col2, col3, col4, col5 = st.columns([1.1, 1.5, 1.5, 1, 1])

        with col1:
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                gridsearch = st.radio('GridSearchCV', ['Add', 'Remove'], index=1, horizontal=True, help="To exhaustively search a parameter grid and evaluate model using cross-validation")
        
        if gridsearch == 'Remove':
            with col2:
                estimators = st.number_input('n_estimators', min_value=50, max_value=1000, value=100, step=50, help="Number of trees in the forest")
            with col3:
                depth = st.number_input('max_depth', min_value=2, max_value=10, value=5, step=1, help="Maximum depth of the tree")
            with col4:
                classweight = st.radio('Class weight', ['Balanced', 'Manual'], index=0, horizontal=True, help="Choose class weight to handle class imbalance")

            if classweight == 'Manual':
                with col5:
                    weight = st.number_input('Enter x (class_weights = {0\: 1, 1: neg/pos * x})', min_value=1.0, max_value=5.0, value=1.0, step=0.5, help="Multiplication factor for the minority class")

                # No Grid Search CV and class_weight = {0: 1, 1: neg / pos * multiplication_factor}
                X_train, X_test, y_train, y_test = prepare_data(data)

                neg, pos = y_train.value_counts()
                scale_pos_weight = neg / pos
                class_weights = {0: 1, 1: scale_pos_weight * weight}

                model = RandomForestClassifier(
                    n_estimators=estimators,  # Number of trees
                    max_depth=depth,      # Limit depth to prevent overfitting
                    random_state=42,
                    class_weight=class_weights  # Custom weights to handle imbalance
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                evaluate_model(y_test, y_pred, y_pred_proba, f"Random Forest ({estimators} est, {depth} dep, neg/pos * {weight})")
        
            else:
                # No Grid Search CV and class_weight = balanced
                X_train, X_test, y_train, y_test = prepare_data(data)

                model = RandomForestClassifier(
                    n_estimators=estimators,  # Number of trees
                    max_depth=depth,      # Limit depth to prevent overfitting
                    random_state=42,
                    class_weight='balanced'  # Handles class imbalance
                )   

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                evaluate_model(y_test, y_pred, y_pred_proba, f'Random Forest ({estimators} est, {depth} dep, balanced weight)')


        else:
            with sub_col2:
                cross_val = st.number_input('cv', min_value=3, max_value=10, value=5, step=1, help="Number of folds in cross-validation")

            with col2:
                est_col1, est_col2, est_col3 = st.columns(3)
                estimator_1 = est_col1.number_input('n_estimators', min_value=50, max_value=1000, value=100, step=50, help="Number of trees in the forest (array of 3 values)")
                estimator_2 = est_col2.number_input('', min_value=50, max_value=1000, value=200, step=50)
                estimator_3 = est_col3.number_input('', min_value=50, max_value=1000, value=300, step=50)

            with col3:
                dep_col1, dep_col2, dep_col3 = st.columns(3)
                depth_1 = dep_col1.number_input('max_depth', min_value=2, max_value=10, value=3, step=1, help="Maximum depth of the tree (array of 3 values)")
                depth_2 = dep_col2.number_input('', min_value=2, max_value=10, value=5, step=1)
                depth_3 = dep_col3.number_input('', min_value=2, max_value=10, value=7, step=1)
            
            with col4:
                classweight = st.radio('Class weight', ["['balanced', 'balanced_subsample']", 'Manual'], index=0, horizontal=True, help="Handle class imbalance by assigning weights")

            # Grid Search CV and class_weight = {0: 1, 1: neg / pos * multiplication_factor}
            if classweight == 'Manual':
                with col5:
                    weight = st.number_input('Enter x (class_weights = {0\: 1, 1: neg/pos * x})', min_value=1.0, max_value=5.0, value=1.0, step=0.5, help="Multiplication factor for the minority class")
                
                
                X_train, X_test, y_train, y_test = prepare_data(data)

                neg, pos = y_train.value_counts()
                scale_pos_weight = neg / pos
                class_weights = {0: 1, 1: scale_pos_weight * weight}

                model = RandomForestClassifier(random_state=42, class_weight='balanced')

                param_grid = {
                    'n_estimators': [estimator_1, estimator_2, estimator_3],
                    'max_depth': [depth_1, depth_2, depth_3],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': [class_weights]
                }

                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cross_val, scoring='f1', n_jobs=-1)
                grid_search.fit(X_train, y_train)

                best_params = grid_search.best_params_
                print(f"Best Parameters: {best_params}")

                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                evaluate_model(y_test, y_pred, y_pred_proba, f"Random Forest ({estimator_1},{estimator_2},{estimator_3} est, {depth_1}, {depth_2}, {depth_3} dep, cv={cross_val}, neg/pos * {weight})")

            # Grid Search CV and class_weight = ['balanced', 'balanced_subsample']
            else:
                X_train, X_test, y_train, y_test = prepare_data(data)
                model = RandomForestClassifier(random_state=42, class_weight='balanced')

                param_grid = {
                    'n_estimators': [estimator_1, estimator_2, estimator_3],
                    'max_depth': [depth_1, depth_2, depth_3],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', 'balanced_subsample']
                }

                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cross_val, scoring='f1', n_jobs=-1)
                grid_search.fit(X_train, y_train)

                best_params = grid_search.best_params_
                print(f"Best Parameters: {best_params}")

                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                evaluate_model(y_test, y_pred, y_pred_proba, f"Random Forest ({estimator_1},{estimator_2},{estimator_3} est, {depth_1}, {depth_2}, {depth_3} dep, cv = {cross_val}, cw [b, b_subsmpl])")    


    # st.write(st.session_state.metrics_df)
    # Define styling function (handles non-numeric values)
    def color_threshold(val):
        if isinstance(val, (int, float)):
            return "color: green" if val > thershold_value else "color: red"
        else:
            return ""  # Skip styling for non-numeric values

    # Increase Styler limit (if needed for large DataFrames)
    pd.set_option("styler.render.max_elements", 1_000_000)

    # Round numeric values to 2 decimal places and format display
    numeric_cols = st.session_state.metrics_df.select_dtypes(include=[np.number]).columns
    styled_df = (
        st.session_state.metrics_df.copy()
        .round(2)  # Round all numeric columns to 2 decimals
        .style
        .format("{:.2f}", subset=numeric_cols)  # Force 2 decimal display
        .applymap(color_threshold, subset=numeric_cols)  # Apply coloring
    )
    st.dataframe(styled_df)

