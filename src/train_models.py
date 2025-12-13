import os
print("Current dir:", os.getcwd())
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Load features
df = pd.read_csv('./url_features.csv')
feature_cols = ['url_length', 'num_digits', 'num_special', 'has_ip', 'path_length', 'domain_length', 'num_subdomains', 'has_suspicious_words', 'entropy']
X = df[feature_cols]
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train models
models = {
    'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
    'naive_bayes': GaussianNB(),
    'random_forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"{name} - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    joblib.dump(model, f'./models/{name}.pkl')

print("Models saved.")