import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# --- Load and clean data ---
df = pd.read_csv(r"C:\Users\jakub\Desktop\CaloriesEstimator\MainCalorieEstimator\data\calories_data.csv")
df = df.drop(columns=["user_id", "user_id.1"], errors="ignore")

# --- Define features and target ---
X = df.drop(columns=["calories", "heart_rate", "body_temp"], errors="ignore")
y = df["calories"]

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# --- XGBoost model configuration ---
model = XGBRegressor(
    n_estimators=800,          # Number of trees
    learning_rate=0.03,        # Learning rate
    max_depth=7,               # Maximum tree depth
    subsample=0.85,            # Fraction of rows used for each tree
    colsample_bytree=0.85,     # Fraction of columns used for each tree
    reg_lambda=2.0,            # L2 regularization strength
    n_jobs=-1                  # Use all CPU cores
)
model.fit(X_train, y_train)

# --- Evaluation ---
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
# mse = mean_squared_error(y_test, pred)
# rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

print("=== XGBoost Evaluation ===")
print(f"MAE : {mae:.3f}")
# print(f"MSE : {mse:.3f}")
# print(f"RMSE: {rmse:.3f}")
print(f"R²  : {r2:.4f}")

# --- Save model (optional) ---
# SAVE_PATH = r"C:\Users\jakub\Desktop\CaloriesEstimator\models\MainCaloriesModel_XGBoost.joblib"
# joblib.dump(model, SAVE_PATH)
# print(f"Model saved to: {SAVE_PATH}")
