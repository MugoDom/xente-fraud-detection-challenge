import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model(model, X_val, y_val, feature_names, output_prefix='validation'):
    # Evaluate the model on the validation set
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Print evaluation metrics
    print(f"Classification Report ({output_prefix}):")
    print(classification_report(y_val, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_val, y_pred_proba))

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{output_prefix}.png')
    plt.close()

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))  # Top 10 features
    plt.title('Top 10 Feature Importance')
    plt.savefig(f'feature_importance_{output_prefix}.png')
    plt.close()

    # Save predictions
    results = pd.DataFrame({
        'actual': y_val,
        'predicted': y_pred,
        'fraud_probability': y_pred_proba
    })
    results.to_csv(f'{output_prefix}_predictions.csv', index=False)
    print(f"Predictions saved to {output_prefix}_predictions.csv")