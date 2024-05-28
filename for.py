import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

st.title("Credit Card Fraud Detection")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")
    return data

data = load_data()

st.subheader("Dataset Shape")
st.write(data.shape)

st.subheader("Dataset Description")
st.write(data.describe())

# Determine the number of fraud cases in the dataset
fraud = len(data[data['Class'] == 1])
valid = len(data[data['Class'] == 0])
outlier_fraction = fraud / (fraud + valid)
st.subheader("Outlier Fraction")
st.write(outlier_fraction)

st.subheader("Class Distribution")
st.write("Fraud Cases:", fraud)
st.write("Valid Transactions:", valid)

# Amount details of fraudulent transaction
st.subheader("Amount details of fraudulent transactions")
st.write(data[data['Class'] == 1]['Amount'].describe())

# Amount details of valid transaction
st.subheader("Amount details of valid transactions")
st.write(data[data['Class'] == 0]['Amount'].describe())

# Correlation matrix
corrmat = data.corr()
st.subheader("Correlation Matrix")
st.write(corrmat)

# Display Correlation matrix as heatmap
st.subheader("Correlation Matrix Heatmap")
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)

# Splitting the dataset
X = data.drop(['Class'], axis=1)
Y = data["Class"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)

# Model evaluation
st.subheader("Model Evaluation")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall:", recall_score(y_test, y_pred))
st.write("F1 Score:", f1_score(y_test, y_pred))
st.write("Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_pred))

# Confusion Matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, y_pred)
st.subheader("Confusion Matrix")
st.write(conf_matrix)

# Visualizing Confusion Matrix
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
st.pyplot(plt)

# Displaying Fraudulent Transactions
st.subheader("Fraudulent Transactions")
fraudulent_transactions = data[data['Class'] == 1]
st.write(fraudulent_transactions)
