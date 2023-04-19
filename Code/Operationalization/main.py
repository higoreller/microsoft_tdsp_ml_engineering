from Model.Treinamento import train
from Operationalization.predictions import get_predictions
import subprocess
import time


def serve_model():
    # If you need to kill process type it on terminal 'fuser -k 1234/tcp'

    # Start the MLflow model server obteined from the train step
    model_uri = "mlruns/796054021220512049/9f2045e593cc4fb5bd237120ec56b423/artifacts/logistic_regression"
    subprocess.Popen(["mlflow", "models", "serve", "--model-uri", model_uri, "--no-conda", "-p", "1234"])

    try:
        # Wait for the server to start
        time.sleep(10)
    except Exception as e:
        print(f"An error occurred: {e}")


def serve_dashboard():
    subprocess.Popen(['streamlit', 'run', 'dashboard.py'])


def main():
    # Train
    train()

    # Start to serve the model
    serve_model()

    # Get predictions
    get_predictions()

    # Start streamlit
    serve_dashboard()


if __name__ == '__main__':
    main()




