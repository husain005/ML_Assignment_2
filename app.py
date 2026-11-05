import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

st.title("ML Assignment 2 - Model Comparison")

uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    X = df.drop("target", axis=1)
    y = df["target"]

    models = {
        "Logistic Regression": "logistic.pkl",
        "Decision Tree": "tree.pkl",
        "KNN": "knn.pkl",
        "Naive Bayes": "nb.pkl",
        "SVM": "svm.pkl"
    }

    results = []

    for model_name, model_file in models.items():

        model = joblib.load(model_file)

        pred = model.predict(X)
        prob = model.predict_proba(X)[:, 1]

        results.append([
            model_name,
            accuracy_score(y, pred),
            roc_auc_score(y, prob),
            precision_score(y, pred),
            recall_score(y, pred),
            f1_score(y, pred),
            matthews_corrcoef(y, pred)
        ])

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ]
    )

    st.subheader("All Model Results")

    st.dataframe(results_df)

    best_model = results_df.sort_values(
        by="Accuracy",
        ascending=False
    ).iloc[0]

    st.success(
        f"Best Model: {best_model['Model']} "
        f"(Accuracy = {best_model['Accuracy']:.4f})"
    )

    selected_model = st.selectbox(
        "Select Model for Confusion Matrix",
        results_df["Model"]
    )

    model_file = models[selected_model]
    model = joblib.load(model_file)

    pred = model.predict(X)

    cm = confusion_matrix(y, pred)

    st.subheader(
        f"Confusion Matrix - {selected_model}"
    )

    fig, ax = plt.subplots(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
