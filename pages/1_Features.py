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
#                                           Features                                            #
#################################################################################################


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
