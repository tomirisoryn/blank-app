import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)




import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Read CSV file
csv_file_path = 'data/Loan_Default 2.csv'
Loan_Default = pd.read_csv(csv_file_path)

# Streamlit app
st.title("Loan Default Analysis")

# Sidebar for user inputs
st.sidebar.header("User Input Features")

# Display the dataset
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(Loan_Default)

# Impute missing values
Loan_Default['Upfront_charges'].fillna(Loan_Default['Upfront_charges'].median(), inplace=True)
Loan_Default['Interest_rate_spread'].fillna(Loan_Default.groupby('loan_type')['Interest_rate_spread'].transform('median'), inplace=True)
Loan_Default['rate_of_interest'].fillna(Loan_Default['rate_of_interest'].median(), inplace=True)
Loan_Default['dtir1'].fillna(Loan_Default['dtir1'].median(), inplace=True)
Loan_Default['income'].fillna(Loan_Default.groupby('Gender')['income'].transform('median'), inplace=True)
Loan_Default['LTV'].fillna(Loan_Default['LTV'].mean(), inplace=True)
Loan_Default['property_value'].fillna(Loan_Default['property_value'].median(), inplace=True)

# Remove columns not useful for analysis
Loan_Default.drop(columns=['ID', 'year', 'Security_Type'], inplace=True)

######Visuals#########

# Credit type default rate visual
st.subheader("Default Rate by Credit Type")
fig1 = px.bar(Loan_Default, x='credit_type', y='Status', color='credit_type', barmode='group',
              labels={'Status': '% Default', 'credit_type': 'Credit Type'},
              title='Default rate by Credit Type')
fig1.update_yaxes(tickformat=".0%")
st.plotly_chart(fig1)

# Average Property amount by Loan Type visual
st.subheader("Property Value by Loan Type")
fig2 = px.bar(Loan_Default, x='loan_type', y='property_value', color='Status', barmode='group',
              labels={'property_value': 'Property Value', 'loan_type': 'Loan Type'},
              title='Property value by Loan Type')
st.plotly_chart(fig2)

# Correlation Matrix
st.subheader("Correlation Matrix")
numeric_columns = Loan_Default.select_dtypes(include=[np.number]).columns
cor_matrix = Loan_Default[numeric_columns].corr()
fig3 = go.Figure(data=go.Heatmap(z=cor_matrix.values, x=cor_matrix.columns, y=cor_matrix.columns, colorscale='Spectral'))
st.plotly_chart(fig3)

######################


# Prepare data for Models
X = Loan_Default.drop(columns=['Status'])
y = Loan_Default['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Handle missing values for categorical variables
categorical_columns = X_train.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    st.write("Handling missing values for categorical variables...")
    # Impute missing values with the most frequent value
    imputer = SimpleImputer(strategy='most_frequent')
    X_train[categorical_columns] = imputer.fit_transform(X_train[categorical_columns])
    X_test[categorical_columns] = imputer.transform(X_test[categorical_columns])

# Encode categorical variables
label_encoders = {}
for column in X_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_train[column] = le.fit_transform(X_train[column])
    X_test[column] = le.transform(X_test[column])
    label_encoders[column] = le

# Handle missing values in X_train and X_test (if any remain)
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)


# Function to get user input
def get_user_input():
    property_value = st.sidebar.number_input("Property Value", min_value=8000, value=16500000)
    loan_type = st.sidebar.selectbox("Loan Type", ["type1", "type2", "type3"])
    credit_type = st.sidebar.selectbox("Credit Type", ["CIB", "CRIF", "EQUI", "EXP"])
    income = st.sidebar.number_input("Income", min_value=0, value=578000)
    credit_score = st.sidebar.number_input("Credit Score", min_value=500, value=900)
    ltv = st.sidebar.number_input("Loan-to-Value Ratio (LTV)", min_value=0.0, value=7831)

    # Create a dictionary from the inputs
    user_data = {
        'property_value': property_value,
        'loan_type': loan_type,
        'credit_type': credit_type,
        'income': income,
        'Credit_Score': credit_score,
        'LTV': ltv
    }

    # Convert the dictionary into a DataFrame
    input_data = pd.DataFrame(user_data, index=[0])
    return input_data

# Get user input
user_input = get_user_input()

# Display user input
st.subheader("User Input:")
st.write(user_input)


# Model 1: Logistic Regression
st.subheader("Logistic Regression Model")
# Train the model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predictions
log_pred = logistic_model.predict(X_test)
accuracy_log = accuracy_score(y_test, log_pred)
st.write(f"Accuracy: {accuracy_log:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, log_pred)
st.write("Confusion Matrix:")
st.write(conf_matrix)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, logistic_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
fig4.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
fig4.update_layout(title='ROC Curve Logistic Regression', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
st.plotly_chart(fig4)

# Model 2: Decision Tree
st.subheader("Decision Tree Model")
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, dt_pred)
st.write(f"Accuracy: {accuracy_dt:.4f}")

# Confusion Matrix
conf_matrix1 = confusion_matrix(y_test, dt_pred)
st.write("Confusion Matrix:")
st.write(conf_matrix1)

# Display Decision Tree Predictions
st.write("Decision Tree Predictions:")
dt_predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': dt_pred
})
st.write(dt_predictions_df)

# Interpret Decision Tree Predictions
st.write("Interpretation of Decision Tree Predictions:")
st.write("- **0**: Loan will **not default**.")
st.write("- **1**: Loan will **default**.")

###### Model 3: Random Forest ######
st.subheader("Random Forest Model")
rf_model = RandomForestClassifier(n_estimators=10, random_state=123)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, rf_pred)
st.write(f"Accuracy: {accuracy_rf:.4f}")


# Encode user input using the same label encoders
for column in categorical_columns:
    if column in user_input.columns:
        user_input[column] = label_encoders[column].transform(user_input[column])

# Ensure the user input has the same columns as the training data
# Add missing columns with default values (0)
for column in X.columns:
    if column not in user_input.columns:
        user_input[column] = 0

# Reorder columns to match the training data
user_input = user_input[X.columns]

# Make a prediction
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

# Display the prediction
st.subheader("Prediction:")
if prediction[0] == 0:
    st.write("The credit will **not default**.")
else:
    st.write("The credit will **default**.")

# Display prediction probabilities
st.subheader("Prediction Probabilities:")
st.write(f"Probability of **No Default**: {prediction_proba[0][0]:.2f}")
st.write(f"Probability of **Default**: {prediction_proba[0][1]:.2f}")


# Confusion Matrix
conf_matrix2 = confusion_matrix(y_test, rf_pred)
st.write("Confusion Matrix:")
st.write(conf_matrix2)

# Model 4: XGBoost
st.subheader("XGBoost Model")
xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=4, random_state=123)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, xgb_pred)
st.write(f"Accuracy: {accuracy_xgb:.4f}")

# Confusion Matrix
conf_matrix3 = confusion_matrix(y_test, xgb_pred)
st.write("Confusion Matrix:")
st.write(conf_matrix3)

