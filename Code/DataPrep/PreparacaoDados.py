import pandas as pd

def load_data(path):
    kobe_data = pd.read_csv(path)
    return kobe_data

def preprocess_data(kobe_data, shot_type):
    # Shot_type can be '2PT Field Goal' or '3PT Field Goal'
    # Load and remove missing data
    kobe_data = kobe_data.dropna()

    # Filtering rows where shot_type is equal to the provided shot_type
    kobe_data = kobe_data[kobe_data['shot_type'] == shot_type]

    # Selecting only the necessary columns
    selected_columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
    kobe_data = kobe_data[selected_columns]

    return kobe_data
