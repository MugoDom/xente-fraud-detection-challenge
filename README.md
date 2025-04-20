# Xente Fraud Detection

This project focuses on detecting fraudulent transactions using supervised machine learning techniques. It leverages real-world transaction data to build a robust classification model that helps distinguish between legitimate and fraudulent financial activity.

## Project Structure
xente-fraud-detection-challenge/
├── data/
│   ├── training.csv
│   ├── test.csv
│   ├── cleaned_training.csv
|   ├── Xente_Variable_Definitions.csv
│   ├── preprocessed_train.csv
│   ├── preprocessed_val.csv
│   └── preprocessed_test.csv

│
├── models/
│   ├── rf model
│   ├── amount_threshold.joblib
│   ├── preprocessor.joblib
│   └── feature_names.joblib
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_rf_tuning.ipynb
│   └── 03_model_evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   └── rf_tuning.py
│
├── plots/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── feature_importance.png
│
└── README.md
