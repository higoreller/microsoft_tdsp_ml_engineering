import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, f1_score
from DataPrep.PreparacaoDados import load_data, preprocess_data
from pycaret.classification import setup, create_model, predict_model, tune_model
import mlflow
import mlflow.sklearn
from mlflow.pyfunc import PythonModel
import numpy as np

class SklearnModelWrapper(PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        labels = self.model.predict(model_input)
        scores = self.model.predict_proba(model_input)[:, 1]  
        return np.column_stack((labels, scores))

class LogisticRegressionWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        labels = self.model.predict(model_input)
        scores = self.model.predict_proba(model_input)[:, 1]
        return np.column_stack((labels, scores))


def split_by_shot_type(test_size, shot_type):
    # Load and preprocess the data
    kobe_data = load_data('./../../Data/Raw/kobe_dataset.csv')
    kobe_data = preprocess_data(kobe_data, shot_type)

    # Save Dataframe to binary
    kobe_data.to_parquet("./../../Data/Processed/data_filtered.parquet")

    # Split the dataset into features (X) and target (y)
    X = kobe_data.drop("shot_made_flag", axis=1)
    y = kobe_data["shot_made_flag"]

    # Split the data into training and test sets in a stratified way
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # Save the training and test sets in parquet files
    X_train.join(y_train).to_parquet("./../../Data/Modeling/base_train.parquet")
    X_test.join(y_test).to_parquet("./../../Data/Modeling/base_test.parquet")

    return X_train, X_test, y_train, y_test


def train(test_size=0.2, shot_type='2PT Field Goal'):
    # Split dataset
    X_train, X_test, y_train, y_test = split_by_shot_type(test_size, shot_type)

    # Initialize MLflow
    mlflow.set_experiment("Training")

    # Set up PyCaret environment
    clf_setup = setup(data=pd.concat([X_train, y_train], axis=1), n_jobs=-2, fold_shuffle=True, log_experiment=True, target="shot_made_flag", experiment_name='Training')

    # Create and tune Logistic Regression model
    lr_model = create_model("lr", cross_validation=True, fold=5, verbose=False)
    lr_tuned = tune_model(lr_model)
    lr_tuned_wrapper = LogisticRegressionWrapper(lr_tuned)
    lr_predicts = predict_model(lr_tuned, data=X_test)

    # Create and tune Random Forest model
    rf_model = create_model("rf", cross_validation=True, fold=5, verbose=False)
    rf_tuned = tune_model(rf_model)
    wrapped_rf_tuned = SklearnModelWrapper(rf_tuned)
    rf_predicts = predict_model(rf_tuned, data=X_test)

    # Log the models and metrics in MLflow
    # Log LogisticRegression model
    mlflow.end_run()
    with mlflow.start_run(run_name="Logistic_Regression"):
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("test_size", len(X_test))
        mlflow.pyfunc.log_model("logistic_regression", python_model=lr_tuned_wrapper)
        mlflow.log_metric("lr_log_loss", log_loss(y_test, lr_predicts['prediction_score']))
        model_uri = mlflow.register_model("runs:/{}/logistic_regression".format(mlflow.active_run().info.run_id), "logistic_regression_model")
        print(model_uri)

    # Log RandomForest model
    with mlflow.start_run(run_name="Random_Forest"):
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("test_size", len(X_test))
        mlflow.pyfunc.log_model("random_forest", python_model=wrapped_rf_tuned)
        mlflow.log_metric("f1_score", f1_score(y_test, rf_predicts["prediction_label"]))
        mlflow.log_metric("rf_log_loss", log_loss(y_test, rf_predicts['prediction_score']))
        model_uri = mlflow.register_model("runs:/{}/random_forest".format(mlflow.active_run().info.run_id), "random_forest_model")
        print(model_uri)

    



