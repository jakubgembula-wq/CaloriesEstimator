import pandas as pd, json, joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

DATA = r"C:\Users\Igor\Downloads\archive (4)\gym_members_exercise_tracking_synthetic_data.csv"
MODEL_PATH = "calories_rf_pipeline.joblib"
SCHEMA_PATH = "calories_rf_schema.json"

df = pd.read_csv(DATA)
obj_cols = df.select_dtypes(include="object").columns
df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

valid_workouts = ["Strength", "Cardio", "HIIT", "Yoga"]
df = df[df["Workout_Type"].isin(valid_workouts)].copy()

df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

for c in ["Age", "Weight (kg)", "Height (m)", "Session_Duration (hours)", "Calories_Burned"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["Calories_Burned"]).reset_index(drop=True)

feature_cols = ["Gender","Age","Weight (kg)","Height (m)","Session_Duration (hours)","Workout_Type"]
TARGET_COL = "Calories_Burned"
cat_cols = ["Workout_Type"]
num_cols = [c for c in feature_cols if c not in cat_cols]

missing = [c for c in feature_cols + [TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}\nGot: {list(df.columns)}")

X, y = df[feature_cols], df[TARGET_COL]

preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ]), cat_cols),
])

pipe = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        n_jobs=-1,
        bootstrap=True
    ))
])

cv = KFold(n_splits=5, shuffle=True)
mae_scores = -cross_val_score(pipe, X, y, scoring="neg_mean_absolute_error", cv=cv)
r2_scores  =  cross_val_score(pipe, X, y, scoring="r2", cv=cv)
print(f"RF 5-fold MAE: mean={mae_scores.mean():.2f} ± {mae_scores.std():.2f}")
print(f"RF 5-fold R^2: mean={r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

pipe.fit(X, y)
joblib.dump(pipe, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")

schema = {"feature_cols": feature_cols, "cat_cols": cat_cols, "num_cols": num_cols}
with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
    json.dump(schema, f, ensure_ascii=False, indent=2)
print(f"Saved schema to {SCHEMA_PATH}")
