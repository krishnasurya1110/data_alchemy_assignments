import streamlit as st
import pandas as pd


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
#                                           Dataset Overview                                    #
#################################################################################################


st.subheader("Problem statement")
st.markdown("""
            Credit risk analysis is crucial for financial institutions, influenced by factors such as applicant demographics, financial behavior, and credit history.
            It is equally important for applicants to understand how their personal finances, housing stability, and credit history impact their ability to repay loans, enabling them to make informed borrowing decisions.
            Identifying key drivers of successful loan repayment is essential for improving risk assessment and lending strategies.
            This project aims to analyze a dataset of applicant profiles and loan characteristics to predict the likelihood of repayment.
            """)

# View dataset
st.markdown("##### Dataset")

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
                        'Target variable (0: loan paid off, 1: loan defaulted)']
        }

cols_df = pd.DataFrame(cols)
st.table(cols_df)
