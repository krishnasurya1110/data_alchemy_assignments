import streamlit as st
import pandas as pd
import plotly.express as px


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
#                                           Sunburst                                            #
#################################################################################################


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
