# # Modification performed on the dataset downloaded from kaggle
# before proceeding with the EDA

import pandas as pd

df = pd.read_csv('loan_dataset_from_kaggle.csv')

# df = pd.read_csv('loan_dataset.csv')
new_column_names = {
                    'person_age': 'age',
                    'person_income': 'income',
                    'person_home_ownership': 'home_ownership',
                    'person_emp_length': 'emp_length',
                    'loan_amnt': 'loan_amount',
                    'cb_person_default_on_file': 'cb_default',
                    'cb_person_cred_hist_length': 'cb_cred_hist',
                    }
df = df.rename(columns=new_column_names)

# Drop outliers
df = df[df['age'] != 123]
df = df[df['emp_length'] != 123]
df = df[df['income'] <= 1000000]

df.to_csv('loan_dataset.csv', index=False)

dff = pd.read_csv('loan_dataset.csv')
print(dff.head())
