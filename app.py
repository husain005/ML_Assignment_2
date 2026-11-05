import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt

st.title("ML Assignment 2 - Classification Models")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "SVM"
    ]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    X = data.drop("target", axis=1)
    y = data["target"]

    if model_name == "Logistic Regression":
        model = joblib.load("logistic.pkl")

    elif model_name == "Decision Tree":
        model = joblib.load("tree.pkl")

    elif model_name == "KNN":
        model = joblib.load("knn.pkl")

    elif model_name == "Naive Bayes":
        model = joblib.load("nb.pkl")

    else:
        model = joblib.load("svm.pkl")

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:,1]

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    auc = roc_auc_score(y, probabilities)
    mcc = matthews_corrcoef(y, predictions)

    st.subheader("Evaluation Metrics")

    st.write("Accuracy:", accuracy)
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1 Score:", f1)
    st.write("AUC:", auc)
    st.write("MCC:", mcc)

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)
