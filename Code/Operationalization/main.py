from Model.Treinamento import train
from DataPrep.PreparacaoDados import load_data
from Operationalization.predictions import get_predictions
import subprocess
import time

kobe_data = load_data('./../../Data/Raw/kobe_dataset.csv')

def serve_model():
    # If you need to kill process type it on terminal 'fuser -k 1234/tcp'
    #subprocess.Popen(["fuser", "-k", "1234/tcp"])
    #time.sleep(1)

    # Start the MLflow model server obteined from the train step
    model_uri = "mlruns/796054021220512049/9f2045e593cc4fb5bd237120ec56b423/artifacts/logistic_regression"
    subprocess.Popen(["mlflow", "models", "serve", "--model-uri", model_uri, "--no-conda", "-p", "1234"])

    try:
        # Wait for the server to start
        time.sleep(3)
    except Exception as e:
        print(f"An error occurred: {e}")


def serve_dashboard():
    subprocess.Popen(['streamlit', 'run', 'dashboard.py'])


def main():
    # Train
    train(kobe_data)

    # Start to serve the model
    serve_model()

    # Get predictions
    get_predictions(kobe_data, '3PT Field Goal')

    # Start streamlit
    serve_dashboard()


if __name__ == '__main__':
    main()




