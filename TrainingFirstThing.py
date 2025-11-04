from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import joblib  # <-- added
import os

INPUT_CSV = r"C:\Users\Igor\Desktop\Calories burned\calories_burned_per_hour_wisconsin_dhs.csv"
SAVE_PATH = r"C:\Users\Igor\Desktop\Calories burned\CaloriesProgram\SecondaryCaloriesModel.joblib"

df = pd.read_csv(INPUT_CSV)

rename_map = {
    "Calories_130_lbs": "Calories_59_kg",
    "Calories_155_lbs": "Calories_70_kg",
    "Calories_190_lbs": "Calories_86_kg"
}
df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

cal_cols = [c for c in ["Calories_59_kg", "Calories_70_kg", "Calories_86_kg"] if c in df.columns]
if len(cal_cols) == 0:
    raise ValueError("No calories columns found. Expected 'Calories_59_kg', 'Calories_70_kg', 'Calories_86_kg'.")

df["MeanCalories"] = df[cal_cols].mean(axis=1)

q1 = df["MeanCalories"].quantile(1/3)
q2 = df["MeanCalories"].quantile(2/3)

def by_calories(val: float) -> str:
    if val < q1:
        return "Low"
    elif val < q2:
        return "Medium"
    else:
        return "High"

df["CalorieDemand"] = df["MeanCalories"].apply(by_calories)

if "Activity" in df.columns:
    df = df.drop(columns=["Activity"])

print("\nCalories-based classes:")
print(df["CalorieDemand"].value_counts().to_string())

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.colheader_justify", "center")

print(df.head())

long_df = df.melt(
    id_vars=["CalorieDemand"],
    value_vars=["Calories_59_kg", "Calories_70_kg", "Calories_86_kg"],
    var_name="WeightCol",
    value_name="Calories"
)

long_df["Weight"] = long_df["WeightCol"].str.extract(r"(\d+)").astype(int)
long_df = long_df[["Weight", "CalorieDemand", "Calories"]]
print(long_df.head(10))

X = pd.get_dummies(long_df[["Weight", "CalorieDemand"]], drop_first=True)
y = long_df["Calories"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nR² (Accuracy): {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f}")

# # === Save the model ===
# os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
# joblib.dump(model, SAVE_PATH)
# print(f"\n Model saved successfully to:\n{SAVE_PATH}")
