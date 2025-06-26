import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def train_model(df_encoded):
    # 1. Separate features and target
    if 'treatment' not in df_encoded.columns:
        raise ValueError("Target variable 'treatment' is missing!")

    X = df_encoded.drop(columns=['treatment'])
    y = df_encoded['treatment']

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # 4. Evaluate model
    y_pred = clf.predict(X_test)
    print("\n✅ Model Training Complete")
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("📉 Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # 5. Save the model
    model_path = os.path.join("outputs", "models", "random_forest_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(clf, model_path)
    print(f"\n💾 Model saved to {model_path}")
    
    # Save column names for future prediction use
    import json
    with open("outputs/models/feature_columns.json", "w") as f:
     json.dump(list(X.columns), f)


    return clf
