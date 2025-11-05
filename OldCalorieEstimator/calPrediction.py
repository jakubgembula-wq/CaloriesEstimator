import json, joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("calories_rf_pipeline.joblib")
SCHEMA_PATH = Path("calories_rf_schema.json")

pipe = joblib.load(MODEL_PATH)
schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
feature_cols = schema["feature_cols"]

def ask(prompt): return input(prompt).strip()

def gender_to_num(g):
    g = g.lower()
    if g in ("male", "m", "0"): return 0
    if g in ("female", "f", "1"): return 1
    return None  

def to_float(x):
    try: return float(x)
    except: return None

def to_int(x):
    try: return int(x)
    except: return None

def parse_height_cm(raw):
    """Convert height from centimeters to meters."""
    v = to_float(raw)
    if v is None:
        return None
    return v / 100.0 

print("\nEnter person data (leave blank to skip):")
gender = gender_to_num(ask("Gender (Male/Female): "))
age = to_int(ask("Age (years): "))
weight = to_float(ask("Weight (kg): "))
height = parse_height_cm(ask("Height (cm): "))
duration_h = to_float(ask("Session duration (hours): "))

wtype_raw = ask("Workout type (Strength/Cardio/HIIT/Yoga): ").strip()
wtype = "HIIT" if wtype_raw.lower() == "hiit" else wtype_raw.title()
if wtype not in {"Strength", "Cardio", "HIIT", "Yoga"}:
    wtype = "Cardio"

row = {
    "Gender": gender,
    "Age": age,
    "Weight (kg)": weight,
    "Height (m)": height,
    "Session_Duration (hours)": duration_h,
    "Workout_Type": wtype,
}

X = pd.DataFrame([row], columns=feature_cols)

est = float(pipe.predict(X)[0])
print(f"\nEstimated calories burned: {est:.1f} kcal")
