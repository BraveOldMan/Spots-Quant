import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

def train_model():
    print("Loading XGBoost features...")
    try:
        df = pd.read_csv("xgboost_features.csv")
    except FileNotFoundError:
        print("xgboost_features.csv not found. Run features.py first.")
        return
        
    # Target: 1 (Home Win), 0 (Draw), -1 (Away Win)
    # We must remap target to 0, 1, 2 for XGBClassifier
    # 0 -> Away Win
    # 1 -> Draw
    # 2 -> Home Win
    mapping = {-1: 0, 0: 1, 1: 2}
    df['target'] = df['target'].map(mapping)
    
    X = df[['elo_diff', 'mom_diff', 'rating_diff', 'mif_home', 'mif_away']]
    y = df['target']
    
    print(f"Dataset shape: {X.shape}")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Define Meta-Learner Model
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        learning_rate=0.05,
        max_depth=4, # 浅层树防止过拟合
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    print("\nRunning Time-Series Cross Validation...")
    accuracies = []
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")
        
    print(f"\nAverage TS-CV Accuracy: {sum(accuracies)/len(accuracies):.4f}")
    
    print("\nTraining Final Model on all historical data...")
    model.fit(X, y)
    
    model.save_model("xgboost_model.json")
    print("Model saved to xgboost_model.json")
    
if __name__ == "__main__":
    train_model()
