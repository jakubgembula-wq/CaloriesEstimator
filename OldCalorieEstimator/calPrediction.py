import json, joblib
import pandas as pd
from pathlib import Path

MODEL_PATH  = Path(r"C:\Users\jakub\Desktop\CaloriesEstimator\OldCalorieEstimator\models\calories_rf_pipeline.joblib")
SCHEMA_PATH = Path(r"C:\Users\jakub\Desktop\CaloriesEstimator\OldCalorieEstimator\models\calories_rf_schema.json")

pipe = joblib.load(MODEL_PATH)

# Load the schema file, parse it into a dictionary, and extract the feature column order
# used during model training.
schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
feature_cols = schema["feature_cols"]   # Column order must remain unchanged.

# Helper function for reading a line of input and trimming extra whitespace.
def ask(question): 
    return input(question).strip()

# Convert text values for gender into numeric codes:
# male → 0, female → 1, other values → None.
def gender_to_num(g):
    g = g.lower()
    if g in ("male", "m", "0"): return 0
    if g in ("female", "f", "1"): return 1
    return None

# Convert text to float; return None if conversion fails.
def to_float(x):
    try: return float(x)
    except: return None

# Convert text to integer; return None if conversion fails.
def to_int(x):
    try: return int(x)
    except: return None

# Convert height from text input in centimeters to meters.
# If conversion to float fails, return None.
def parse_height_cm(raw):
    v = to_float(raw)
    if v is None:
        return None
    return v / 100.0

# Collect required input values from the user.
# Missing or invalid values are converted to None.
print("\nEnter person data (leave blank to skip):")

gender     = gender_to_num(ask("Gender (Male/Female): "))
age        = to_int(ask("Age (years): "))
weight     = to_float(ask("Weight (kg): "))
height     = parse_height_cm(ask("Height (cm): "))
duration_h = to_float(ask("Session duration (hours): "))

# Normalize workout type:
# - Map any form of "hiit" to "HIIT"
# - Convert other values to title-case
# - If not in the allowed set, default to "Cardio"
wtype_raw = ask("Workout type (Strength/Cardio/HIIT/Yoga): ")
wtype = "HIIT" if wtype_raw.lower() == "hiit" else wtype_raw.title()
if wtype not in {"Strength", "Cardio", "HIIT", "Yoga"}:
    wtype = "Cardio"

# Build a single data row matching the exact field names
# used during model training.
row = {
    "Gender": gender,
    "Age": age,
    "Weight (kg)": weight,
    "Height (m)": height,
    "Session_Duration (hours)": duration_h,
    "Workout_Type": wtype,
}

# Convert the row into a DataFrame.
# The 'columns=feature_cols' argument ensures correct column order.
X = pd.DataFrame([row], columns=feature_cols)

# Generate a prediction using the pipeline (preprocessing included).
est = float(pipe.predict(X)[0])

# Display the predicted calorie value, formatted to one decimal place.
print(f"\nEstimated calories burned: {est:.1f} kcal")
