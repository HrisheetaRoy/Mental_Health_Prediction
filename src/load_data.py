import pandas as pd

def load_data(filepath="data/survey.csv"):
    df = pd.read_csv(filepath)
    return df
