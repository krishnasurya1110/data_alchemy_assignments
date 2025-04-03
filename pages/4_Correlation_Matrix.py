import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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
#                                    Correlation Matrix                                         #
#################################################################################################

st.write("")
st.write("##### Correlation Matrix Tables (after label encoding `loan_grade`)")

df_1 = df.copy()
df_1['cb_default'] = df_1['cb_default'].map({'N': 0, 'Y': 1})

label_encoder = LabelEncoder()
df_1['loan_grade'] = label_encoder.fit_transform(df_1['loan_grade'])

numerical_columns_1 = [col for col in df_1.columns if df_1[col].dtype in ['int64', 'float64'] and col != 'id']
df_1_numerical = df_1[numerical_columns_1]
corr_matrix_1 = df_1_numerical.corr()

st.table(corr_matrix_1.style.background_gradient(cmap="coolwarm").format("{:.2f}"))

with st.expander("**Findings**"):
    st.markdown("""
                - **Inferences:**
                    - From this correlation plot, we can verify that higher the `loan_grade`, higher the `loan_int_rate`. But since their correlation is as high as 0.94, we can proceed to drop loan_grade and keep only loan_int_rate to avoid multicollinearity.
                    - `age` and `cb_cred_hist` having a high correlation of 0.88. It is obvious that higher the age, the longer their credit histories would be. We can drop the age columnm as well for aforementioned reasons.

                - **Other observations:**
                    - `cb_default` has a correlation of 0.55 with loan_grade and 0.50 with loan_interest_rate. This means that the loans for defaulters are usually associated with high risk. Hence, the higher interest rate and grade.
                    - `loan_amount` has a high correlation of 0.65 with loan_percent_income indicating that higher the loan amount, the larger it tends to be of a person's income.
                    - `loan_amount` has a positive correlation of 0.34 with person_income. Indicating that higher income people tend to take larger loans.

                - **Correlations with target variable (loan_status):**
                    - `loan_grade` (0.39) and `loan_interest_rate` (0.34): indicates that riskier loans associated with higher interest rates have higher chances of loan being defaulted.
                    - `loan_percent_income` (0.38): higher the commitment of income to loan, higher the chances of default.
                    - `cb_default` (0.19): indicates that people who defaulted on a loan in the past are more likely to default again.
                    - `loan_amount` (0.14): indicates that people who take larger loans are more likely to default.
                    - `person_income` (-0.18) and `emp_length` (-0.18): indicates that people with higher incomes/ longer employment lengths are less likely to default.
                """)

df_1.drop(['age', 'loan_grade'], axis=1, inplace=True)


st.text("")
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

plt.close()

with st.expander("**Findings**"):
    st.markdown("""
                - **Key inferences:**
                    - There isn't much difference among the correlations various types of `loan_intent`.
                    - However, the relationship exhibited by two `home_ownership` categories RENT and MORTGAGE are quite interesting.
                - **Home Ownership:**
                    - `income` and `emp_length`:
                        - MORTGAGE: income (0.30) and emp_length (0.29) shows that mortgage seems to be the most common mode of housing ownership for people with higher income and those who are well into their career.
                        - RENT: income (-0.28) and emp_length (-0.29) shows that renting is more common among people with lower income and those who are just starting their careers.

                    - `loan_int_rate`, `loan_percent_income`, `cb_default`:
                        - MORTGAGE: loan_int_rate (-0.20), loan_percent_income (-0.16), cb_default (-0.10) shows that people with mortgage tend to have lower interest rates, lower percentage of income committed to loans, and are less likely to default.
                        - RENT: loan_int_rate (0.20), loan_percent_income (0.15), cb_default (0.24) shows that people with rent tend to have higher interest rates, higher percentage of income committed to loans, and are more likely to default.
                
                    - Target varibale (`loan_status`):
                        - MORTGAGE: loan_status (-0.20) shows that people with mortgage are less likely to default.
                        - RENT:loan_status (0.24) shows that people with rent are more likely to default.
                """)
