# Xente Fraud Detection Challenge

## Overview
This project is a solution to the Xente Fraud Detection Challenge, where the goal is to build a machine learning model to predict fraudulent transactions in the Xente dataset. The project implements a complete machine learning pipeline, including data preprocessing, feature engineering, model training, and evaluation. The final model is a Random Forest classifier, and the pipeline handles class imbalance using SMOTE and preprocesses data using a combination of numerical scaling, categorical encoding, and custom feature engineering.

The pipeline is modular and split into five main notebooks:
1. **data_exploration**: Explores the data to get an understanding of the dataset.
2. **data_preprocessing**: Cleans the data, engineers features, and preprocesses the training, validation, and test datasets.
3. **model_training**: Trains a Random Forest model on the preprocessed training data.
4. **model_evaluation**: Evaluates the model on the validation set and generates predictions for the test set.
5. **hyperparameter_tuining**: Tunes the model using Gridsearch CV.

## Project Structure
```
xente-fraud-detection-challenge/
├── data/
│   ├── training.csv        # Raw training data with FraudResult labels
│   └── test.csv            # Raw test data for predictions
├── models/
│   ├── amount_threshold.joblib    # Amount threshold for feature engineering
│   ├── feature_names.joblib       # Feature names after preprocessing
│   ├── preprocessor.joblib        # Fitted preprocessor
│   ├── random_forest_model.joblib # Trained Random Forest model
│   ├── rf.joblib                  # Alternative Random Forest model (possibly untuned)
│   └── rf_model_tuned.joblib      # Tuned Random Forest model
├── notebooks/
│   ├── .ipynb_checkpoints/        # Jupyter notebook checkpoints
│   ├── 01_data_exploration.ipynb  # Notebook for data exploration
│   ├── 02_data_preprocessing.ipynb # Notebook for data preprocessing
│   ├── 03_model_training.ipynb    # Notebook for model training
│   ├── 04_evaluation_and_tuning.ipynb # Notebook for evaluation and tuning
│   ├── 05_hyperparameter_tuning.ipynb # Notebook for hyperparameter tuning
│   ├── confusion_matrix_validation.png # Confusion matrix plot (output)
│   ├── feature_importance_validation.png # Feature importance plot (output)
│   ├── validation_predictions.csv    # Validation predictions (output)
│   └── test_predictions.csv          # Test predictions (output)
├── plots/
│   # (Empty directory for storing plots)
├── src/
│   ├── __pycache__/               # Python cache files
│   ├── data_preprocessing.py      # Preprocessing pipeline functions
│   ├── model_evaluation.py        # Model evaluation functions
│   ├── model_training.py          # Model training functions
│   └── model_tuning.py            # Model tuning functions
├── tests/
│   # (Empty directory for tests, currently unused)
├── venv/
│   # (Virtual environment directory)
├── .gitignore                     # Git ignore file
└── README.md                      # Project documentation
```

## Prerequisites
To run this project, you’ll need Python 3.10 or later and the following dependencies:

- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn` (for SMOTE)
- `matplotlib` (for plotting)
- `joblib`

You can install the dependencies using the following command:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib joblib
```

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd xente-fraud-detection-challenge
   ```

2. **Prepare the Data**:
   - Place the raw data files (`training.csv` and `test.csv`) in the `data/` directory.
   - Ensure the data files are in the correct format:
     - `training.csv`: Should include columns like `TransactionId`, `BatchId`, `AccountId`, `SubscriptionId`, `CustomerId`, `CurrencyCode`, `CountryCode`, `ProviderId`, `ProductId`, `ProductCategory`, `ChannelId`, `Amount`, `Value`, `TransactionStartTime`, `PricingStrategy`, and `FraudResult`.
     - `test.csv`: Same columns as `training.csv` but without the `FraudResult` column.

3. **Set Up the Directory Structure**:
   - Ensure the `src/` directory contains the following scripts: `preprocess_data_notebook.py`, `train_model_notebook.py`, `evaluate_model_notebook.py`, `preprocessing_pipeline.py`, `model_training.py`, and `model_evaluation.py`.

## Running the Pipeline
The pipeline is split into three stages: preprocessing, training, and evaluation. Run the scripts in the following order from the `src/` directory.

### 1. Preprocessing
The `data_processing.py` script cleans the raw data, performs feature engineering, and preprocesses the training, validation, and test datasets. It also saves the preprocessed data and preprocessing objects.

```bash
cd src
python data_processing.py
```

**Outputs**:
- `cleaned_training.csv`: Cleaned training data.
- `preprocessed_train.csv`: Preprocessed training data.
- `preprocessed_val.csv`: Preprocessed validation data.
- `preprocessed_test.csv`: Preprocessed test data.
- `amount_threshold.joblib`: Amount threshold for high-amount flag feature.
- `preprocessor.joblib`: Fitted preprocessing pipeline.
- `feature_names.joblib`: Feature names after preprocessing.

### 2. Training
The `model_training.py` script loads the preprocessed training data and trains a Random Forest model.

```bash
python model_training.py
```

**Outputs**:
- `random_forest_model.joblib`: Trained Random Forest model.

### 3. Evaluation and Prediction
The `model_evaluation.py` script evaluates the model on the validation set and generates predictions for the test set.

```bash
python model_evaluation.py
```

**Outputs**:
- `validation_predictions.csv`: Predictions on the validation set with actual labels, predicted labels, and fraud probabilities.
- `test_predictions.csv`: Predictions on the test set with predicted labels and fraud probabilities.
- `confusion_matrix_validation.png`: Confusion matrix plot for the validation set.
- `feature_importance_validation.png`: Feature importance plot for the validation set.
- Console output: Classification report and ROC AUC score for the validation set.

## Feature Engineering
The preprocessing pipeline includes the following feature engineering steps:
- **Datetime Features**: Extract `hour`, `day_of_week`, and `month` from `TransactionStartTime`.
- **Log Transformation**: Apply log transformation to `Amount` to handle skewness (`log_amount`).
- **High-Risk Flags**:
  - `high_amount_flag`: Flag transactions above the 90th percentile of `Amount`.
  - `high_fraud_provider`: Flag transactions from high-fraud providers (`ProviderId_1`, `ProviderId_3`, `ProviderId_5`).
  - `high_fraud_channel`: Flag transactions from high-fraud channels (`ChannelId_1`, `ChannelId_3`, `ChannelId_2`).
  - `high_fraud_category`: Flag transactions in high-fraud categories (`transport`, `utility_bill`, `financial_services`).

## Preprocessing Steps
- **Dropped Columns**: Irrelevant ID columns (`TransactionId`, `BatchId`, etc.) and `Value` (due to high correlation with `Amount`).
- **Numerical Features**: Scaled using `StandardScaler` after imputing missing values with the mean.
- **Categorical Features**: One-hot encoded using `OneHotEncoder` after imputing missing values with `'missing'`.
- **Binary Features**: Passed through unchanged.
- **Class Imbalance**: Handled using SMOTE on the training data.

## Model
- **Algorithm**: Random Forest Classifier.
- **Hyperparameters**: Default parameters are used (can be tuned in `model_tuning.py`).
- **Training Data**: Preprocessed training data with SMOTE applied to balance the classes.

## Evaluation Metrics
The model is evaluated on the validation set using:
- Classification Report (precision, recall, F1-score).
- ROC AUC Score.
- Confusion Matrix (visualized as a plot).
- Feature Importance (visualized as a plot).

## Notes
- **Data Quality**: Ensure there are no `NaN` values in the `FraudResult` column of `training.csv`. The pipeline includes checks and imputation to handle missing values in other columns.
- **Scalability**: The pipeline is designed for the Xente dataset but can be adapted for other fraud detection tasks by modifying the feature engineering and preprocessing steps.
- **Future Improvements**:
  - Experimentation with other algorithms (e.g., XGBoost, LightGBM).
  - Additional feature engineering based on domain knowledge.

## License
This project is licensed under the MIT License.

## Acknowledgments
- The Xente Fraud Detection Challenge dataset[https://zindi.africa/competitions/xente-fraud-detection-challenge].
- The `scikit-learn` and `imbalanced-learn` libraries for machine learning tools.