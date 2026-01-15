from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

def train_model(train_X, train_y, val_X, val_y):
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(train_X, train_y)
    
    # Validation
    preds = model.predict(val_X)
    print("\nValidation Results:")
    print(classification_report(val_y, preds))
    
    return model

def save_model(model, path='weather_model.joblib'):
    joblib.dump(model, path)
    print(f"Model saved to {path}")
