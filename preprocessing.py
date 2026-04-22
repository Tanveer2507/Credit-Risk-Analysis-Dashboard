import pandas as pd

def load_data(path='german_credit_data.csv'):
    df = pd.read_csv(path)   # ✅ FIXED
    return df

def create_target(df):
    # Create synthetic risk (for demo purpose)
    # High credit + long duration = high risk
    df['Risk'] = ((df['Credit amount'] > 5000) & (df['Duration'] > 24)).astype(int)
    return df

def preprocess_data(df):
    # Drop unnecessary column
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Create target
    df = create_target(df)

    # Handle missing values
    df.fillna('Unknown', inplace=True)

    # Encode categorical columns
    df = pd.get_dummies(df, drop_first=True)

    return df