import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# Store encoder for reuse between train and test
ordinal_encoder = None

def preprocess_data(df, is_train=True):
    """
    Feature engineering and preprocessing using OrdinalEncoder.
    
    Parameters:
        df (pd.DataFrame): Input data.
        is_train (bool): Whether this is training or test data.
    
    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series or None): Target if is_train=True, else None.
    """
    global ordinal_encoder
    df = df.copy()

    # --- Time features ---
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['hour'] = df['TransactionStartTime'].dt.hour
    df['dayofweek'] = df['TransactionStartTime'].dt.dayofweek

    # --- Log transforms ---
    df['log_amount'] = np.log1p(df['Amount'])
    df['log_value'] = np.log1p(df['Value'])

    # --- Risk features ---
    high_risk_providers = ['ProviderId_1', 'ProviderId_3', 'ProviderId_5']
    high_risk_channels = ['ChannelId_1', 'ChannelId_3', 'ChannelId_2']
    risky_categories = ['transport', 'utility_bill', 'financial_services']

    df['high_risk_provider'] = df['ProviderId'].isin(high_risk_providers).astype(int)
    df['high_risk_channel'] = df['ChannelId'].isin(high_risk_channels).astype(int)
    df['risky_category'] = df['ProductCategory'].isin(risky_categories).astype(int)

    # --- Amount flags ---
    df['large_amount'] = (df['Amount'] > 10000).astype(int)
    df['amount_to_value_ratio'] = df['Amount'] / (df['Value'] + 1)

    # --- Drop unnecessary columns ---
    drop_cols = [
        'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
        'TransactionStartTime', 'Amount', 'Value', 'CurrencyCode'
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # --- Encode categoricals ---
    categorical_cols = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']
    if is_train:
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[categorical_cols] = ordinal_encoder.fit_transform(df[categorical_cols])
    else:
        df[categorical_cols] = ordinal_encoder.transform(df[categorical_cols])

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