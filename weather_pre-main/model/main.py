from preprocess import preprocess
from train import train_model, save_model
from sklearn.metrics import classification_report

def run_pipeline():
    print("--- Starting Weather Prediction Pipeline ---")
    
    # 1. Preprocess Data
    print("Loading and preprocessing data...")
    train_X, train_y, val_X, val_y, test_X, test_y = preprocess()
    
    # 2. Train Model
    model = train_model(train_X, train_y, val_X, val_y)
    
    # 3. Final Evaluation on Test Set
    print("\n--- Final Test Set Evaluation ---")
    test_preds = model.predict(test_X)
    print(classification_report(test_y, test_preds))
    
    # 4. Save artifacts
    save_model(model)
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
