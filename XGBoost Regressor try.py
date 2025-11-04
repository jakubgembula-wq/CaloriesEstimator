import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# --- Load & clean ---
df = pd.read_csv(r"C:\Users\Igor\Desktop\Calories burned\calories_data.csv")
df = df.drop(columns=["user_id", "user_id.1"], errors="ignore")

# --- Features & target (same logic as before: drop HR & Temp) ---
X = df.drop(columns=["calories", "heart_rate", "body_temp"], errors="ignore")
y = df["calories"]

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- XGBoost model (strong defaults / “overkill” but stable) ---
model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# --- Evaluate ---
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

print("=== XGBoost Evaluation ===")
print(f"MAE : {mae:.3f}")
print(f"MSE : {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²  : {r2:.4f}")

# --- Save model ---
SAVE_PATH = r"C:\Users\Igor\Desktop\Calories burned\CaloriesProgram\MainCaloriesModel_XGBoost.joblib"
joblib.dump(model, SAVE_PATH)
print(f"Model saved to: {SAVE_PATH}")
