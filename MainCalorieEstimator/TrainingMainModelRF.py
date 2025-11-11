import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

data = pd.read_csv(r"C:\Users\jakub\Desktop\CaloriesEstimator\MainCalorieEstimator\data\calories_data.csv")

# Remove identifier columns if they are present.
if "user_id" in data.columns:
    data = data.drop(columns=["user_id"])
if "user_id.1" in data.columns:
    data = data.drop(columns=["user_id.1"])

print("Dataset loaded successfully!")
print(f"Rows: {len(data)}, Columns: {list(data.columns)}\n")
print("Preview of dataset:")
print(data.head(10).to_string(index=False))
print("\n")

# Define the target column and remove unused sensor fields.
target_col = "calories"
heartRate = "heart_rate"
BodyTemperature = "body_temp"
X = data.drop(columns=[target_col, heartRate, BodyTemperature])
y = data[target_col]

# Split data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Initialize the RandomForest regressor.
model = RandomForestRegressor(
    n_estimators=300,
    n_jobs=-1
)

# Train the model.
model.fit(X_train, y_train)

# Generate predictions for the test data.
pred = model.predict(X_test)

# Evaluate model performance.
mae = mean_absolute_error(y_test, pred)
# mse = mean_squared_error(y_test, pred)
# rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

print("=== Evaluation ===")
print(f"MAE : {mae:.3f}")
print(f"R²  : {r2:.4f}")

# Calculate and display feature importance scores.
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n=== Feature Importance ===")
print(importances.to_string())

# Show a sample of true vs predicted values.
preview = pd.DataFrame({
    "y_true": y_test.values[:10],
    "y_pred": pred[:10]
})
print("\n=== Sample predictions (first 10) ===")
print(preview.to_string(index=False))

# Save the model if needed.
# joblib.dump(model, r"C:\Users\jakub\Desktop\CaloriesEstimator\models\MainCaloriesModel.joblib")
# print("\nModel saved to: C:\\Users\\jakub\Desktop\\CaloriesEstimator\\models\MainCaliersModel.joblib")
