from Model.Treinamento import train
from DataPrep.PreparacaoDados import load_data
from Operationalization.predictions import get_predictions
import subprocess
import time
import mlflow
from mlflow.tracking import MlflowClient


kobe_data = load_data('./../../Data/Raw/kobe_dataset.csv')


def extract_relative_path(model_uri, base_path):
    return model_uri.replace(base_path, '')


def get_latest_lr_model_uri(experiment_name):
    base_path = "file:///home/higoreller/Development/pos_ml_engineering/Code/Operationalization/"
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"])
    logistic_regression_model_found = False

    for run in runs:
        if run.info.status == "FINISHED":
            if not logistic_regression_model_found:
                logistic_regression_model_found = True
            else:
                model_uri = run.info.artifact_uri + "/logistic_regression"
                relative_model_uri = extract_relative_path(model_uri, base_path)
                return relative_model_uri
    
    raise ValueError("No successful model training run found")


def serve_model():
    # If you need to kill process type it on terminal 'fuser -k 1234/tcp'
    subprocess.Popen(["fuser", "-k", "1234/tcp"])
    time.sleep(1)

    # Start the MLflow model server obteined from the train step
    experiment_name = "Training"
    model_uri = get_latest_lr_model_uri(experiment_name)
    print(model_uri)
    #model_uri = "mlruns/796054021220512049/9f2045e593cc4fb5bd237120ec56b423/artifacts/logistic_regression"
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




