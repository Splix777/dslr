import json
from typing import Any, Dict, List

import pandas as pd
import numpy as np
import streamlit as st
import os

from logreg_train import LogisticRegressionTrainer
from logreg_predict import LogisticRegressionPredictor


def get_project_base_path() -> str:
    """Get the base path of the project."""
    current_file = os.path.abspath(__file__)
    return os.path.abspath(os.path.join(current_file, "..", "..", ".."))


def train_page():
    st.title("Logistic Regression Trainer")

    # Sidebar controls
    st.sidebar.header("Training Parameters")
    num_iterations = st.sidebar.slider(
        "Number of Iterations", 100, 5000, 2000, 100
    )
    learning_rate = st.sidebar.selectbox(
        "Learning Rate", [0.001, 0.01, 0.1, 0.5, 1.0]
    )
    epsilon_option = st.sidebar.selectbox("Epsilon", [1e-8, 1e-7, 1e-6, 1e-5])

    # Train model
    trainer = LogisticRegressionTrainer(
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        epsilon=epsilon_option,
    )

    if dataset_path := st.file_uploader("Upload Dataset (CSV)", type="csv"):
        data = trainer.load_data(dataset_path)
        feature_names = data.columns[5:]
        st.session_state.feature_names = feature_names

    if "results" not in st.session_state:
        st.session_state.results = None

    # Train model
    if st.button("Train Model"):
        st.session_state.results = trainer.train_one_vs_all()
        st.write("Model trained successfully!")

    # Display results (e.g., charts, tables)
    if st.button("Show Results"):
        with st.spinner("Loading results..."):
            display_results(
                st.session_state.results, st.session_state.feature_names
            )


def display_results(results: List[Dict[str, Any]], feature_names: List[str]):
    st.subheader("Training Results")

    for result in results:
        st.markdown(f"# House: {result['house']}")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Final Weights:**")
            weights_df = pd.DataFrame(
                {"Feature": feature_names, "Weight": result["weights"]}
            )
            weights_df.set_index("Feature", inplace=True)
            st.dataframe(weights_df, height=530)

        with col2:
            st.write("**Final Bias:**")
            st.write(result["bias"])

        st.write("**Costs over Iterations:**")
        st.line_chart(result["costs"])

        st.write("**Weights over Iterations:**")
        weight_history_df = pd.DataFrame(
            result["weight_history"], columns=feature_names
        )
        st.line_chart(weight_history_df)

        st.write("**Bias over Iterations:**")
        st.line_chart(result["bias_history"])


def predict_page():
    st.title("Prediction Page")
    # Check if the model has been trained and weight file exists
    base_dir = get_project_base_path()
    logreg_dir = os.path.join(base_dir, "outputs/logistic_regression")
    csv_dir = os.path.join(base_dir, "csv_files")
    weights_file = os.path.join(logreg_dir, "weight_base.json")
    csv_test_file = os.path.join(csv_dir, "dataset_test.csv")

    if not os.path.exists(weights_file) or not os.path.exists(csv_test_file):
        st.warning("Please train the model first!")
        return

    st.write("### Example of test file before prediction")
    test_data = pd.read_csv(csv_test_file, nrows=5)
    st.dataframe(test_data, hide_index=True)

    predictor = LogisticRegressionPredictor(weights_file)
    test_data = predictor.load_data(csv_test_file)
    predicted_labels = predictor.predict(test_data)

    st.write("### After Inserting the Predicted Labels to the Test Data")
    test_data["Hogwarts House"] = predicted_labels
    st.dataframe(test_data.head(5), hide_index=True)

    true_csv_file = os.path.join(csv_dir, "sample_truth.csv")
    true_csv = pd.read_csv(true_csv_file)
    true_labels = true_csv["Hogwarts House"].values

    st.write("### Comparison of True and Predicted Labels")
    comparison_df = pd.DataFrame(
        {"True Labels": true_labels, "Predicted Labels": predicted_labels}
    )

    df = pd.DataFrame(true_labels)
    df2 = pd.DataFrame(predicted_labels)

    st.dataframe(df.compare(df2), width=800, key="compare")

    # evaluation_file = os.path.join(logreg_dir, "evaluation.txt")
    # predictor.evaluate_model(
    #     true_labels, np.array(predicted_labels), evaluation_file
    # )
    # with open(evaluation_file, "r") as f:
    #     evaluation_results = f.read()
    #     st.write(evaluation_results)


def main():
    st.set_page_config(layout="wide")

    pages = {"Train Model": train_page, "Predict": predict_page}

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    page = pages[selection]
    page()


# Run Streamlit app
if __name__ == "__main__":
    main()
