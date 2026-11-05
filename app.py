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

# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("ML Assignment 2 - Model Comparison")

st.write(
    """
    Upload the test CSV file containing all input features
    and the target column.
    The application evaluates all saved models without
    retraining them.
    """
)

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

# --------------------------------------------------
# PROCESS CSV
# --------------------------------------------------

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    # Check target column

    if "target" not in df.columns:
        st.error(
            "ERROR: Uploaded CSV must contain a column named 'target'."
        )
        st.stop()

    X = df.drop("target", axis=1)
    y = df["target"]

    # Expected number of features for Breast Cancer Dataset

    expected_features = 30

    if X.shape[1] != expected_features:
        st.error(
            f"""
            ERROR:
            Expected {expected_features} input features,
            but found {X.shape[1]}.
            """
        )
        st.stop()

    # --------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------

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

        predictions = model.predict(X)

        probabilities = model.predict_proba(X)[:, 1]

        accuracy = accuracy_score(y, predictions)

        auc = roc_auc_score(y, probabilities)

        precision = precision_score(y, predictions)

        recall = recall_score(y, predictions)

        f1 = f1_score(y, predictions)

        mcc = matthews_corrcoef(y, predictions)

        results.append([
            model_name,
            round(accuracy, 4),
            round(auc, 4),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            round(mcc, 4)
        ])

    # --------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------

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

    st.subheader("Model Performance Comparison")

    st.dataframe(
        results_df,
        use_container_width=True
    )

    # --------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------

    best_model = results_df.sort_values(
        by="Accuracy",
        ascending=False
    ).iloc[0]

    st.success(
        f"""
        Best Model: {best_model['Model']}
        | Accuracy = {best_model['Accuracy']}
        """
    )

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    st.subheader("Confusion Matrix")

    selected_model = st.selectbox(
        "Select Model",
        list(models.keys())
    )

    model = joblib.load(
        models[selected_model]
    )

    predictions = model.predict(X)

    cm = confusion_matrix(
        y,
        predictions
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")

    ax.set_ylabel("Actual")

    ax.set_title(
        f"{selected_model} Confusion Matrix"
    )

    st.pyplot(fig)

    # --------------------------------------------------
    # CLASSIFICATION SUMMARY
    # --------------------------------------------------

    st.subheader("Assignment Checklist")

    st.success("✓ CSV Uploaded")
    st.success("✓ Saved Models Loaded")
    st.success("✓ No Model Retraining")
    st.success("✓ Metrics Calculated")
    st.success("✓ Results for All Models Displayed")
    st.success("✓ Confusion Matrix Generated")
