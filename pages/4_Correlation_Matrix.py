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
#                                           Correlation Matrix                                  #
#################################################################################################

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
