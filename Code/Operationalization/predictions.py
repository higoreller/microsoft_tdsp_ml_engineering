import pandas as pd
import requests
import json
from DataPrep.PreparacaoDados import load_data, preprocess_data
from sklearn.metrics import log_loss, f1_score

def get_predictions():
    # Define API endpoint
    api_url = "http://127.0.0.1:1234/invocations"

    # Load and preprocess new data
    new_data = load_data('./../../Data/Raw/kobe_dataset.csv')
    new_data = preprocess_data(new_data, '3PT Field Goal')

    # Remove the 'shot_made_flag' column before sending the data
    new_data_without_label = new_data.drop(columns=['shot_made_flag'])

    # Send POST request to API with the new data
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({"dataframe_records": new_data_without_label.to_dict(orient='records')})
    response = requests.post(api_url, headers=headers, data=payload)

    # Check the response JSON
    response_json = json.loads(response.content.decode('utf-8'))

    # Get the predictions from the response
    predictions = pd.DataFrame(response_json['predictions'], columns=['prediction_label', 'prediction_score'])

    # Calculate log loss and f1 score
    log_loss_new = log_loss(new_data['shot_made_flag'], predictions['prediction_score'])
    f1_score_new = f1_score(new_data['shot_made_flag'], predictions['prediction_label'])

    # Print results
    print("New log loss:", log_loss_new)
    print("New f1 score:", f1_score_new)
    print(predictions.head())

    return predictions
