import pandas as pd
import joblib
import json

def load_model(path="outputs/models/random_forest_model.pkl"):
    return joblib.load(path)

def load_sample():
    """
    Define a new sample with some default values.
    This must match  the structure of the encoded training data.
    We'll align column order after loading expected column names.
    """
    sample = {
    'Age': 38,
    'Gender_female': 1,
    'Gender_male': 0,
    'Gender_other': 0,
    
    'benefits_Yes': 1,              # No mental health benefits
    'care_options_Yes': 2,          # No care options provided
    'wellness_program_Yes': 0,      # No wellness program
    'remote_work_Yes': 0,           # Cannot work remotely
    'tech_company_Yes': 1,          # Works in tech (often high-stress)
    
    'seek_help_Yes': 1,             # Not encouraged to seek help
    'anonymity_Yes': 1,             # Anonymity not protected
    'leave_Very difficult': 1,      # Hard to take mental health leave
    'mental_health_consequence_Yes': 1,  # Believes there will be consequences
    'phys_health_consequence_Yes': 0,    # Physical health not perceived as risky
    'coworkers_Yes': 0,             # Can't talk to coworkers about mental health
    'supervisor_Yes': 0,            # Can't talk to supervisor either
    'mental_health_interview_Yes': 1,    # Wouldn't talk about it in an interview
    'mental_vs_physical_Worse': 1,       # Thinks mental health is treated worse
    'obs_consequence_Yes': 1        # Thinks mental health affects job
}

    
    
#     sample = {
#     'Age': 34,
#     'Gender_female': 0,
#     'Gender_male': 1,
#     'Gender_other': 0,
#     'benefits_Yes': 0,
#     'care_options_Yes': 1,
#     'wellness_program_Yes': 0,
#     'remote_work_Yes': 0,
#     'tech_company_Yes': 0,
#     # Other columns will be added/zeroed automatically
# }
    
    
    
    
#     sample = {
#         'Age': 28,
#         'Gender_female': 1,
#         'Gender_male': 0,
#         'Gender_other': 0,
#         'benefits_Yes': 1,
#         'care_options_Yes': 0,
#         'wellness_program_Yes': 1,
#         'remote_work_Yes': 1,
#         'tech_company_Yes': 1,
#         # Other columns will be added/zeroed automatically
#     }

    df = pd.DataFrame([sample])
    return df

def load_expected_columns(path="outputs/models/feature_columns.json"):
    with open(path, "r") as f:
        return json.load(f)

def main():
    model = load_model()
    expected_cols = load_expected_columns()
    
    new_data = load_sample()
    
    # Align with training columns (fill missing with 0)
    new_data = new_data.reindex(columns=expected_cols, fill_value=0)

    prediction = model.predict(new_data)[0]
    label = "Needs treatment" if prediction == 1 else "No treatment needed"

    print(f"\n Prediction: {label} (class: {prediction})")

if __name__ == "__main__":
    main()
