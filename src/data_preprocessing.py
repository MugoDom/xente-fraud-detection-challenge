import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

def preprocess_data(data_path, is_training=True):
    # Step 1: Load data
    data = pd.read_csv(data_path)

    # Step 2: Initial Cleaning
    # Drop 'Value' due to high correlation with 'Amount' (0.99)
    data = data.drop(columns=['Value'])

    # Drop irrelevant ID columns
    drop_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
    data = data.drop(columns=drop_cols)

    # Step 3: Feature Engineering
    # Convert TransactionStartTime to datetime and extract features
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    data['hour'] = data['TransactionStartTime'].dt.hour
    data['day_of_week'] = data['TransactionStartTime'].dt.dayofweek
    data['month'] = data['TransactionStartTime'].dt.month
    data = data.drop(columns=['TransactionStartTime'])

    # Log-transform Amount to handle skewness (add small constant to avoid log(0))
    data['log_amount'] = np.log1p(data['Amount'].abs() + 1)

    # Flag high-amount transactions (above 90th percentile)
    amount_threshold = data['Amount'].quantile(0.9)
    data['high_amount_flag'] = (data['Amount'] > amount_threshold).astype(int)

    # Flag high-fraud providers (ProviderId 1, 3, 5)
    data['high_fraud_provider'] = data['ProviderId'].isin(['ProviderId_1', 'ProviderId_3', 'ProviderId_5']).astype(int)

    # Flag high-fraud channels (ChannelId 1, 3, 2)
    data['high_fraud_channel'] = data['ChannelId'].isin(['ChannelId_1', 'ChannelId_3', 'ChannelId_2']).astype(int)

    # Flag high-fraud categories (transport, utility, financial services)
    high_fraud_categories = ['transport', 'utility_bill', 'financial_services']
    data['high_fraud_category'] = data['ProductCategory'].isin(high_fraud_categories).astype(int)

    # Step 4: Define features and target (if training data)
    if is_training:
        # Drop rows with NaN target values before splitting
        data = data.dropna(subset=['FraudResult'])
        data['FraudResult'] = pd.to_numeric(data['FraudResult'], errors='coerce')
        X = data.drop('FraudResult', axis=1)
        y = data['FraudResult']

    else:
        X = data
        y = None

    # Step 5: Split data (if training data)
    if is_training:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = None, X, None, None

    # Step 6: Preprocessing Pipeline
    # Define numerical and categorical columns
    numerical_cols = ['Amount', 'log_amount', 'hour', 'day_of_week', 'month']
    categorical_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    binary_cols = ['high_amount_flag', 'high_fraud_provider', 'high_fraud_channel', 'high_fraud_category']

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
            ('bin', 'passthrough', binary_cols)
        ])

    # Fit and transform data
    if is_training:
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        # Step 7: Handle Class Imbalance with SMOTE (only for training data)
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
    else:
        # For test data, only transform (preprocessor must be fitted already)
        X_train_processed, y_train_balanced = None, None
        X_test_processed = preprocessor.transform(X_test)

    # Step 8: Get feature names for the processed data
    feature_names = (
        numerical_cols +
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)) +
        binary_cols
    )

    return (X_train_balanced if is_training else None, 
            y_train_balanced if is_training else None, 
            X_test_processed, 
            y_test if is_training else None, 
            feature_names, 
            preprocessor)