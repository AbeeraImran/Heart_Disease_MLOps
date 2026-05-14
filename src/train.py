import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix
import joblib

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:../mlruns")
mlflow.set_experiment("Heart Disease Prediction")

# Load dataset
df = pd.read_csv("data/heart.csv")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Print shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)




from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train Logistic Regression
lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

# Predictions
lr_predictions = lr_model.predict(X_test)

# Accuracy
lr_accuracy = accuracy_score(y_test, lr_predictions)


with mlflow.start_run(run_name="Logistic Regression"):

    # Parameters
    mlflow.log_param("model", "LogisticRegression")

    # Train model
    lr_model = LogisticRegression()

    lr_model.fit(X_train, y_train)

    # Predictions
    lr_predictions = lr_model.predict(X_test)

    # Accuracy
    lr_accuracy = accuracy_score(y_test, lr_predictions)

    # Log metrics
    mlflow.log_metric("accuracy", lr_accuracy)

    # Log model
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")

    print("Logistic Regression Accuracy:", lr_accuracy)


print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))



from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test)

# Accuracy
rf_accuracy = accuracy_score(y_test, rf_predictions)

with mlflow.start_run(run_name="Random Forest"):

    # Parameters
    mlflow.log_param("model", "RandomForestClassifier")

    # Train model
    rf_model = RandomForestClassifier(random_state=42)

    rf_model.fit(X_train, y_train)

    # Predictions
    rf_predictions = rf_model.predict(X_test)

    # Accuracy
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    # Log metrics
    mlflow.log_metric("accuracy", rf_accuracy)

    # Log model
    mlflow.sklearn.log_model(rf_model, "random_forest_model")

    print("Random Forest Accuracy:", rf_accuracy)

    run = mlflow.active_run()
    model_name = "HeartDiseaseModel"

    mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/random_forest_model",
        name=model_name
    )


print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions))


# Save models
joblib.dump(lr_model, "models/logistic_regression_model.pkl")

joblib.dump(rf_model, "models/random_forest_model.pkl")

print("\nModels saved successfully.")



from retrain_logic import evaluate_and_promote

# After all training and registration is done
if __name__ == "__main__":
    evaluate_and_promote()