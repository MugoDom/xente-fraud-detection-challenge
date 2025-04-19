import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Dictionary to store encoders
label_encoders = {}

def preprocess_data(df, is_train=True):
    """
    This function performs feature engineering and preprocessing.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame (train or test).
        is_train (bool): Whether this is training data (includes target).
        
    Returns:
        X (pd.DataFrame): Processed features.
        y (pd.Series): Target (if is_train=True), else None.
    """
    
    df = df.copy()  # Create a copy of the original dataset
    
    # --- Time features ---
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['hour'] = df['TransactionStartTime'].dt.hour
    df['dayofweek'] = df['TransactionStartTime'].dt.dayofweek

    # --- Log transforms ---
    df['log_amount'] = np.log1p(df['Amount'])
    df['log_value'] = np.log1p(df['Value'])
    
    # --- Risk flags ---
    high_risk_providers = ['ProviderId_1', 'ProviderId_3', 'ProviderId_5']
    high_risk_channels = ['ChannelId_1', 'ChannelId_3', 'ChannelId_2']
    risky_categories = ['transport', 'utility_bill', 'financial_services']

    df['high_risk_provider'] = df['ProviderId'].isin(high_risk_providers).astype(int)
    df['high_risk_channel'] = df['ChannelId'].isin(high_risk_channels).astype(int)
    df['risky_category'] = df['ProductCategory'].isin(risky_categories).astype(int)

    # --- Amount flags ---
    df['large_amount'] = (df['Amount'] > 10000).astype(int)
    df['amount_to_value_ratio'] = df['Amount'] / (df['Value'] + 1)

    # --- Drop unneeded columns ---
    drop_cols = [
        'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
        'TransactionStartTime', 'Amount', 'Value', 'CurrencyCode'
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # --- Encode categoricals ---
    categorical_cols = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']
    for col in categorical_cols:
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            df[col] = le.transform(df[col])

    # --- Features & target ---
    features = [
        'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId',
        'PricingStrategy', 'hour', 'dayofweek',
        'log_amount', 'log_value', 'amount_to_value_ratio',
        'high_risk_provider', 'high_risk_channel', 'risky_category', 'large_amount'
    ]

    X = df[features]
    y = df['FraudResult'] if is_train else None

    return X, y
