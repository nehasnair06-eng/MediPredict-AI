import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


# =====================================================
# DIABETES MODEL
# =====================================================

print("\n========== DIABETES PREDICTION ==========\n")

diabetes = pd.read_csv("data/diabetes.csv")

X = diabetes.drop("Outcome", axis=1)
y = diabetes["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Models
diabetes_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


best_model = None
best_accuracy = 0
best_model_name = ""


for name, model in diabetes_models.items():

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    print(name)
    print("Accuracy:", round(accuracy, 4))
    print()


    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name


print("Best Diabetes Model:", best_model_name)
print("Best Accuracy:", round(best_accuracy, 4))


# Save best model
joblib.dump(best_model, "models/diabetes_model.pkl")
joblib.dump(scaler, "models/diabetes_scaler.pkl")

print("\nDiabetes model saved successfully!")


# =====================================================
# HEART DISEASE MODEL
# =====================================================

print("\n========== HEART DISEASE PREDICTION ==========\n")

heart = pd.read_csv("data/heart.csv")

print("Heart Dataset Shape:", heart.shape)

X = heart.drop("target", axis=1)
y = heart["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


heart_scaler = StandardScaler()

X_train_scaled = heart_scaler.fit_transform(X_train)
X_test_scaled = heart_scaler.transform(X_test)


# Models
heart_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


best_heart_model = None
best_heart_accuracy = 0
best_heart_model_name = ""


for name, model in heart_models.items():

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    print(name)
    print("Accuracy:", round(accuracy, 4))
    print()


    if accuracy > best_heart_accuracy:
        best_heart_accuracy = accuracy
        best_heart_model = model
        best_heart_model_name = name


print("Best Heart Disease Model:", best_heart_model_name)
print(
    "Best Accuracy:",
    round(best_heart_accuracy, 4)
)


# Detailed report
heart_predictions = best_heart_model.predict(X_test_scaled)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        heart_predictions
    )
)


# Save model
joblib.dump(
    best_heart_model,
    "models/heart_model.pkl"
)

joblib.dump(
    heart_scaler,
    "models/heart_scaler.pkl"
)


print("\nHeart disease model saved successfully!")