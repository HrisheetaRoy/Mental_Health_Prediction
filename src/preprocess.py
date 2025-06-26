import pandas as pd

def clean_data(df):
    """
    Clean and prepare the mental health dataset.
    - Drop irrelevant columns
    - Handle missing values
    - Encode binary categorical variables
    """
    
    # ----------------------------
    # 1. Drop unnecessary columns
    # ----------------------------
    drop_cols = ['Timestamp', 'state', 'comments']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # ----------------------------
    # 2. Drop rows with missing target
    # ----------------------------
    df = df[df['treatment'].notnull()]

    # ----------------------------
    # 3. Fill missing values (basic strategy)
    # ----------------------------
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # ----------------------------
    # 4. Encode target label
    # ----------------------------
    df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0})

    # ----------------------------
    # 5. Optional: Drop outliers (already done in EDA, but double check)
    # ----------------------------
    df = df[(df['Age'] >= 16) & (df['Age'] <= 70)]

    return df


def encode_features(df):
    """
    Encode categorical features using one-hot encoding.
    """
    cat_cols = df.select_dtypes(include=['object']).columns

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df_encoded
