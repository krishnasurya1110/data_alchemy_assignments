import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Set page configuration to wide mode
st.set_page_config(layout="wide", page_title="Loan Prediction", page_icon="📊",)

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

st.title("Loan Approval Prediction 📊")

@st.cache_data
def load_data():
    df = pd.read_csv('loan_dataset.csv')
    return df

df = load_data()

#################################################################################################
#                                           Datatypes                                           #
#################################################################################################


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
