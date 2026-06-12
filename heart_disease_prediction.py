import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
data = pd.read_csv("heart.csv")

# Display first 5 rows
print("Dataset Preview:")
print(data.head())

# Features and Target
X = data.drop("target", axis=1)
y = data["target"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train Model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict New Patient Data
new_patient = [[52, 1, 2, 130, 220, 0, 1, 150, 0, 1.5, 1, 0, 2]]

prediction = model.predict(new_patient)

if prediction[0] == 1:
    print("\nPrediction: Heart Disease Detected")
else:
    print("\nPrediction: No Heart Disease")
