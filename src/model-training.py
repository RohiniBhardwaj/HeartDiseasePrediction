import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv("data/heart.csv")


# Features and target
X = df.drop("condition", axis=1)
y = df["condition"]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Try multiple ML models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

best_model = None
best_accuracy = 0

print("\nModel Results:")
print("----------------------")

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc:.4f}")

    # Track best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

print(f"\nBest Model: {type(best_model).__name__} with accuracy {best_accuracy:.4f}")

# 5. Save best model and scaler
joblib.dump(best_model, "models/best_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("\n✔ Model and scaler saved!")
