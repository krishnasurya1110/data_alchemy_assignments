import streamlit as st

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


#################################################################################################
#                                           Insights                                            #
#################################################################################################

st.write("")

# st.write("##### Core findings")

# st.title("🔍 Loan Data Insights Dashboard")

st.subheader("📌 Analytical Insights")

eda_insights = [
    "🏠 **Home Ownership Trends**: Renting and mortgages are the most common housing choices among loan applicants.",
    "🎓 **Loan Intent Insights**: Education loans are the most frequently taken, while home improvement loans are the least.",
    "📉 **Loan Risk Distribution**: Lower-risk loans (A, B, C) dominate, whereas high-risk loans (D and above) are fewer in number.",
    "✅ **Default History**: The majority of applicants (about 5:1) have no prior defaults, indicating a more creditworthy applicant pool.",
    "💳 **Repayment vs. Default**: Most loans are successfully repaid, except for high-risk categories (D to G), where defaults are more common.",
    "💵 **Income Insights**: Most applicants earn under \$200K, with very few making over \$800K. High-income earners tend to take larger loans.",
    "📊 **Loan Amount & Interest**: Loan amounts and interest rates appear to be randomly distributed, but higher amounts and interest rates correlate with defaults.",
    "💼 **Loan-to-Income Ratio**: Most loans take up less than 40% of the borrower's income, with the majority falling under 20%.",
    "📉 **Credit History & Default Risk**: Longer employment and higher income reduce default risk, while lower income and shorter employment show higher default peaks.",
    "🔄 **Correlation Findings**: Loan grade, interest rates, and past defaults strongly predict future defaults, while higher income and employment length lower the risk."
]

for insight in eda_insights:
    st.write(insight)

st.subheader("📌 Prediction Insights")

pred_insights = [
    "📈 The **imbalance** in data is addressed using the SMOTE technique, which generates synthetic samples for the minority class. This helps in improving the model's performance by providing a more balanced dataset for training.",
    "🎯 Although **Logistic Regression, KNN and ADABoosting models** perform well in terms of accuracy, they seem to struggle a bit with the precision and recall metrics. This indicates that while they are good at identifying the majority class, they may not be as effective in identifying the minority class (i.e., predicting loan defaults).",
    "⚖️ **XGBoost and Random Forest models**, on the other hand, show a better balance between precision and recall. This suggests that they are more effective in identifying both classes, making them more reliable for predicting loan defaults.",
    "🤝 They also have the highest AUC scores, indicating their superior performance in successfully differentiating the **likelihood of a loan being paid back/ defaulted.**",
]

for insight in pred_insights:
    st.write(insight)
