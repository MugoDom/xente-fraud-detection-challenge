from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train, model_path='random_forest_model.joblib'):
    # Train a Random Forest model
    model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',  # this helps address class imbalance
    max_depth=None,           # defaults can be tuned further with GridSearchCV
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1
)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model