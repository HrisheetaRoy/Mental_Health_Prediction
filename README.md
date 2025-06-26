# Mental Health in Tech Survey Analysis

This project analyzes mental health trends in the tech industry using survey data. It includes a complete pipeline for exploratory data analysis (EDA), data preprocessing, model training, evaluation, and prediction. The goal is to identify whether a person is likely to seek mental health treatment based on their responses.

## Project Structure

Mental_Health_Analysis/
│
├── data/ # Raw dataset (e.g. survey.csv)
├── outputs/
│ └── models/ # Trained model (.pkl) and feature list
│ └──figures/
├── src/
│ ├── init.py
│ ├── eda1.ipynb # Exploratory Data Analysis
│ ├── load_data.py # Loads the dataset
│ ├── preprocess.py # Data cleaning and encoding
│ ├── model.py # Model training and evaluation
│
├── main.py # Runs the full pipeline
├── predict.py # Loads model and predicts on new input
└── README.md


---

## Features

- EDA notebook to explore distributions, missing values, and outliers
- Clean and modular Python scripts for each stage of the pipeline
- Preprocessing handles:
  - Dropping irrelevant columns
  - Missing value imputation
  - One-hot encoding
- Trains a Random Forest classifier to predict mental health treatment likelihood
- Evaluation includes accuracy, precision, recall, F1-score, and confusion matrix
- Saves trained model as `.pkl` for future use
- Predicts on new data using saved model and features

---

## Setup Instructions

### 1. Clone the Repository
 https://github.com/HrisheetaRoy/Mental_Health_Prediction.git
 cd Mental_Health_Analysis

### 2. Create and Activate Virtual Environment

python -m venv .venv  
Activate (Windows)
.venv\Scripts\activate
Activate (Mac/Linux)
source .venv/bin/activate


### 3. Install Dependencies

pip install -r requirements.txt


---

## Running the Project

### 1. Run Full Pipeline (Data Load → Clean → Train)

python main.py


This will:
- Load and preprocess the survey data
- Train a Random Forest model
- Save the model to `outputs/models/random_forest_model.pkl`
- Save the feature column order to `feature_columns.json`

### 2. Run a Prediction on New Data

python predict.py


This will:
- Load the trained model and expected features
- Predict the mental health treatment likelihood for a manually defined sample

---

## Dataset

The dataset used in this project comes from [Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey). It contains responses from tech workers about mental health support in their workplaces.

---

## Model Details

- **Classifier**: Random Forest
- **Target variable**: `treatment` (binary: Yes/No)
- **Encoding**: One-hot encoding for categorical features
- **Train/Test Split**: 80/20
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

## Example Output

✅ Data loaded and preprocessed successfully.
🧮 Final shape: (1250, 135)

✅ Model Training Complete

Classification Report:
precision recall f1-score support
0 0.81 0.79 0.80 115
1 0.82 0.84 0.83 135

Accuracy: 0.82


---

## Future Work

- Hyperparameter tuning with GridSearchCV
- Streamlit web interface for live predictions
- Model interpretability using SHAP or LIME
- Time-based filtering to compare trends across years

---

## Author

This project is built by Hrisheeta Roy as part of a data science portfolio.  
For suggestions, collaborations, or issues, feel free to open a GitHub issue or reach out.

