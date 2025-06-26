import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from .src.load_data import load_data
from .src.preprocess import clean_data, encode_features
# from model import train_model  ← uncomment after model.py is ready


    
    
from .src.model import train_model

def main():
    df = load_data("data/survey.csv")
    df_clean = clean_data(df)
    df_encoded = encode_features(df_clean)

    print("✅ Data loaded and preprocessed successfully.")
    print("🧮 Final shape:", df_encoded.shape)

    model = train_model(df_encoded)

if __name__ == "__main__":
     main()

