import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

data = pd.read_csv(r"C:\Users\jakub\Desktop\CaloriesEstimator\data\calories_data.csv")

if "user_id" in data.columns:
    data = data.drop(columns=["user_id"])
if "user_id.1" in data.columns:
    data = data.drop(columns=["user_id.1"])

print("Dataset loaded successfully!")
print(f"Rows: {len(data)}, Columns: {list(data.columns)}\n")
print("Preview of dataset:")
print(data.head(10).to_string(index=False))
print("\n")

target_col = "calories"
heartRate = "heart_rate"
BodyTemperature = "body_temp"
X = data.drop(columns=[target_col,heartRate,BodyTemperature])
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = RandomForestRegressor(
    n_estimators=300,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

print("=== Evaluation ===")
print(f"MAE : {mae:.3f}")
print(f"MSE : {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²  : {r2:.4f}")

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n=== Feature Importance ===")
print(importances.to_string())

preview = pd.DataFrame({
    "y_true": y_test.values[:10],
    "y_pred": pred[:10]
})
print("\n=== Sample predictions (first 10) ===")
print(preview.to_string(index=False))

# Save the model (optional)
joblib.dump(model, r"C:\Users\jakub\Desktop\CaloriesEstimator\models\MainCaloriesModel.joblib")
print("\nModel saved to: C:\\Users\\jakub\Desktop\\CaloriesEstimator\\models\MainCaliersModel.joblib")
