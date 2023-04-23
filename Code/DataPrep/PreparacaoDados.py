import pandas as pd

def load_data(path):
    data = pd.read_csv(path)
    return data

def preprocess_data(data, shot_type):
    # Check if data is a DataFrame or not
    if not isinstance(data, pd.DataFrame):
        data = pd.read_csv(data)
        
    # Load and remove missing data
    data = data.dropna()

    # Filtering rows where shot_type is equal to the provided shot_type
    if 'shot_type' in data.columns:
        # Filtering rows where shot_type is equal to the provided shot_type
        data = data[data['shot_type'] == shot_type]
    else:
        raise KeyError("The 'shot_type' column is missing from the input DataFrame. Please ensure that the input data contains the 'shot_type' column.")

    # Selecting only the necessary columns
    selected_columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
    data = data[selected_columns]

    return data
