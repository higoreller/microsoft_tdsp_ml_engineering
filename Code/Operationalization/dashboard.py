import pandas as pd
from DataPrep.PreparacaoDados import preprocess_data
from sklearn.metrics import log_loss, f1_score
import streamlit as st
from Operationalization.predictions import get_predictions


def monitor():
    st.title("Kobe Bryant Shot Monitoring")

    st.header("Load data")
    uploaded_file = st.file_uploader("Select csv file", type=["csv"])

    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        new_data = preprocess_data(new_data, '3PT Field Goal')
        new_data_without_label = new_data.drop(columns=['shot_made_flag'])

        st.header("Loaded data")
        st.write(new_data_without_label.head())

        if st.button("Calculate predictions"):
            predictions = get_predictions()

            log_loss_new = log_loss(new_data['shot_made_flag'], predictions['prediction_score'])
            f1_score_new = f1_score(new_data['shot_made_flag'], predictions['prediction_label'])

            st.header("Results")
            st.write("Log loss:", log_loss_new)
            st.write("F1 score:", f1_score_new)
            st.write(predictions.head())


if __name__ == '__main__':
    monitor()