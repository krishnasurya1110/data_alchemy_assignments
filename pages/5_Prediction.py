import streamlit as st
import pandas as pd
import numpy as np

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
#                                           Prediction                                          #
#################################################################################################

df_1 = df.copy()
df_1['cb_default'] = df_1['cb_default'].map({'N': 0, 'Y': 1})

# Label Encoding
label_encoder = LabelEncoder()
df_1['loan_grade'] = label_encoder.fit_transform(df_1['loan_grade'])
df_1.drop(['age', 'loan_grade'], axis=1, inplace=True)

# One-Hot Encoding
df_2 = df_1.copy()

one_hot_encoder = OneHotEncoder(sparse_output=False)  # Keep all categories
columns_to_encode = ['loan_intent', 'home_ownership']
encoded_features = one_hot_encoder.fit_transform(df_2[columns_to_encode])
encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(columns_to_encode), index=df_2.index)
df_encoded = df_2.drop(columns=columns_to_encode).join(encoded_df)

st.write("")

# Display the encoded dataframe
data = df_encoded.copy()
st.write("##### Dataframe used for modeling")
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
    thershold_value = st.number_input('Choose a threshold to color cells', min_value=0.50, max_value=1.0, value=0.70, step=0.05, help="Values great than this will be colored green and less than this will be colored red")

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
        return "color: green" if val >= thershold_value else "color: red"
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
