import pandas as pd
from DataPrep.PreparacaoDados import preprocess_data
from sklearn.metrics import log_loss, f1_score
import streamlit as st
from Operationalization.predictions import get_predictions
from Model.Treinamento import train


def monitor():
    st.title("Kobe Bryant Shot Monitoring")
    st.markdown("---")

    st.header("Load data")
    uploaded_file = st.file_uploader("Select csv file", type=["csv"])

    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)

        with st.form("Set train parameters"):
            col1, col2 = st.columns(2)
            test_size =col1.slider("Set test size", min_value=0.01, max_value=0.5, value=0.01)
            shot_type = col2.select_slider("Set filter option", options=("2PT Field Goal", "3PT Field Goal"))
            form_submit = st.form_submit_button("Train")

            if form_submit:
                progress_bar = st.progress(0)
                def update_progress(progress):
                    progress_bar.progress(progress)


                train_results = train(new_data, test_size=test_size, shot_type=shot_type, progress_bar=update_progress)
                st.text(train_results)
                st.markdown("---")
                st.header("Loaded data")
                new_data_processed = preprocess_data(new_data, shot_type)
                new_data_without_label = new_data_processed.drop(columns=['shot_made_flag'])
                st.dataframe(new_data_without_label.head())
            
                st.markdown("---")
                st.header("Predictions")
                predictions = get_predictions(new_data, shot_type)
                log_loss_new = log_loss(new_data_processed['shot_made_flag'], predictions['prediction_score'])
                f1_score_new = f1_score(new_data_processed['shot_made_flag'], predictions['prediction_label'])
                st.header("Results")
                st.write("Log loss:", log_loss_new)
                st.write("F1 score:", f1_score_new)
                st.write(predictions.head())            

        
if __name__ == '__main__':
    monitor()