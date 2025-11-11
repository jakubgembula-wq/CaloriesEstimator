import pandas as pd, json, joblib
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# --- Paths / constants -------------------------------------------------------
BASE = Path(__file__).resolve().parent            # Directory of this file.
DATA = BASE / "data" / "gym_members_exercise_tracking_synthetic_data.csv"  # Path to the input CSV file.
MODEL_PATH = "calories_rf_pipeline.joblib"       # Output path for the trained pipeline.
SCHEMA_PATH = "calories_rf_schema.json"          # Output path for the schema information.

# --- Load data ---------------------------------------------------------------
df = pd.read_csv(DATA)  # Load the dataset.

# Clean string columns by removing leading/trailing whitespace.
# Step 1: find all object (string) columns.
# Step 2: trim whitespace from each value in these columns.
obj_cols = df.select_dtypes(include="object").columns
df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

# Keep only supported workout types.
# Rows with unsupported values are removed to maintain consistent categories.
valid_workouts = ["Strength", "Cardio", "HIIT", "Yoga"]
df = df[df["Workout_Type"].isin(valid_workouts)].copy()  # .copy() avoids SettingWithCopy warnings.

# Convert Gender to numeric codes:
# Male → 0, Female → 1. Other values become NaN for later imputation if needed.
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# Convert selected columns to numeric.
# Invalid values become NaN for imputation.
for c in ["Age", "Weight (kg)", "Height (m)", "Session_Duration (hours)", "Calories_Burned"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows where the target value (Calories_Burned) is missing.
df = df.dropna(subset=["Calories_Burned"]).reset_index(drop=True)

# --- Feature / target specification -----------------------------------------
# Feature columns used as input (X).
feature_cols = ["Gender","Age","Weight (kg)","Height (m)","Session_Duration (hours)","Workout_Type"]
# Target column (y).
TARGET_COL = "Calories_Burned"

# Split features into numeric and categorical groups.
# Numeric: median imputation.
# Categorical: most-frequent imputation + one-hot encoding.
cat_cols = ["Workout_Type"]
num_cols = [c for c in feature_cols if c not in cat_cols]

# Verify that all required columns exist in the dataset.
missing = [c for c in feature_cols + [TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}\nGot: {list(df.columns)}")

# Separate the features (X) and target (y).
X, y = df[feature_cols], df[TARGET_COL]

# --- Preprocessing pipeline
# Preprocessing steps:
# - Numeric columns: median imputation.
# - Categorical columns: most-frequent imputation + one-hot encoding.
#   handle_unknown="ignore" allows unseen categories during prediction.
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ]), cat_cols),
])

# Create a pipeline combining preprocessing and a RandomForestRegressor.
pipe = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestRegressor(
        n_estimators=300,           # Number of trees in the forest.
        n_jobs=-1,                  # Use all available CPU cores.
    ))
])

# --- Cross-validation evaluation
# 5-fold cross-validation with shuffling.
# The model is trained on 4 folds and evaluated on the remaining fold, repeated 5 times.
cv = KFold(n_splits=5, shuffle=True)

mae_scores = -cross_val_score(pipe, X, y, scoring="neg_mean_absolute_error", cv=cv)
r2_scores  =  cross_val_score(pipe, X, y, scoring="r2", cv=cv)

print(f"RF 5-fold MAE: mean={mae_scores.mean():.2f} ± {mae_scores.std():.2f}")
print(f"RF 5-fold R^2: mean={r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

# --- Train final model on full data -----------------------------------------
# Train the pipeline on the full dataset.
# During fitting:
# - Preprocessing learns imputation values and category mappings.
# - The model learns patterns to predict calorie values.
pipe.fit(X, y)

# joblib.dump(pipe, MODEL_PATH)  # Save the complete pipeline to a file.
# print(f"Saved model to {MODEL_PATH}")

# schema = {"feature_cols": feature_cols, "cat_cols": cat_cols, "num_cols": num_cols}
# with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
#     json.dump(schema, f, ensure_ascii=False, indent=2)
# print(f"Saved schema to {SCHEMA_PATH}")

# Cross-validation reference results:
# RF 5-fold MAE: mean=272.80 ± 10.47
# RF 5-fold R^2: mean=-0.0532 ± 0.0453
