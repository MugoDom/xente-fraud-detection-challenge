import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

def main():
    # Load training data
    train_df = pd.read_csv('../data/preprocessed_train.csv')
    X_train = train_df.drop('FraudResult', axis=1)
    y_train = train_df['FraudResult']

    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    # Initialize RandomForest model
    rf = RandomForestClassifier(random_state=42)

    # Set up GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        verbose=2,
        n_jobs=-1
    )

    # Fit grid search on training data
    grid_search.fit(X_train, y_train)

    # Print the best parameters
    print("Best Hyperparameters:", grid_search.best_params_)

    # Save the best tuned model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, '../models/rf_model_tuned.joblib')

    # Load validation data for evaluation
    val_df = pd.read_csv('../data/preprocessed_val.csv')
    X_val = val_df.drop('FraudResult', axis=1)
    y_val = val_df['FraudResult']

    # Make predictions and compute metrics
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]

    # Print classification report and ROC AUC score
    print("\nClassification Report (Validation):")
    print(classification_report(y_val, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_val, y_proba))

    # Save predictions
    predictions_df = pd.DataFrame({'Predictions': y_pred})
    predictions_df.to_csv('../data/validation_predictions.csv', index=False)

if __name__ == "__main__":
    main()
