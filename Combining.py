import joblib
import pandas as pd

# Load models
rf_model = joblib.load(r"C:\Users\Igor\Desktop\Calories burned\CaloriesProgram\MainCaloriesModel.joblib")
lr_model = joblib.load(r"C:\Users\Igor\Desktop\Calories burned\CaloriesProgram\SecondaryCaloriesModel.joblib")

# Trust slider
RF_WEIGHT = 0.8

# Inputs (no HR/temp)
gender = int(input("Gender (1=male, 0=female): "))
age = int(input("Age: "))
height = float(input("Height (cm): "))
weight = float(input("Weight (kg): "))
duration_min = float(input("Duration (minutes): "))
demand = input("Demand (Low/Medium/High): ").strip().capitalize()

# RF input: only 5 features
rf_input = pd.DataFrame([{
    "gender": gender,
    "age": age,
    "height": height,
    "weight": weight,
    "duration": duration_min,
}])

# LR one-hot (High is baseline => zeros)
lr_input = pd.DataFrame([{
    "Weight": int(round(weight)),
    "CalorieDemand_Low": 1 if demand == "Low" else 0,
    "CalorieDemand_Medium": 1 if demand == "Medium" else 0
}])

# Predict
rf_total = float(rf_model.predict(rf_input)[0])
duration_hours = max(duration_min, 1e-6) / 60.0
rf_rate = rf_total / duration_hours

lr_hourly = float(lr_model.predict(lr_input)[0])
blended_rate = RF_WEIGHT * rf_rate + (1 - RF_WEIGHT) * lr_hourly
blended_total = blended_rate * duration_hours

print(f"{blended_total:.2f}")
