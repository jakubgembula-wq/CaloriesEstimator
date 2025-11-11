from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import os

# ---- CONFIG ----
RF_WEIGHT = 0.8
RF_PATH  = r"C:\Users\jakub\Desktop\CaloriesEstimator\MainCalorieEstimator\models\MainCaloriesModel.joblib"
XGB_PATH = r"C:\Users\jakub\Desktop\CaloriesEstimator\MainCalorieEstimator\models\MainCaloriesModel_XGBoost.joblib"
LR_PATH  = r"C:\Users\jakub\Desktop\CaloriesEstimator\MainCalorieEstimator\models\SecondaryCaloriesModelNameBased.joblib"

for p in (RF_PATH, XGB_PATH, LR_PATH):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model file missing: {p}")

# ---- LOAD MODELS ----
rf_model  = joblib.load(RF_PATH)
xgb_model = joblib.load(XGB_PATH)
lr_model  = joblib.load(LR_PATH)

app = Flask(__name__)

def predict_all(gender, age, height, weight, duration_min, demand, rf_weight=RF_WEIGHT):
    """Return RF-only, XGB-only, LR-only, RFxLR, XGBxLR totals (kcal)."""
    # Common pieces
    features = {
        "gender": int(gender),
        "age": float(age),
        "height": float(height),
        "weight": float(weight),
        "duration": float(duration_min),
    }
    duration_hours = max(float(duration_min), 1e-6) / 60.0  # avoid /0

    # --- RF total ---
    rf_input = pd.DataFrame([features])
    rf_total = float(rf_model.predict(rf_input)[0])  # total kcal (trained that way)
    rf_rate  = rf_total / duration_hours            # kcal/h

    # --- XGB total ---
    xgb_input = pd.DataFrame([features])
    xgb_total = float(xgb_model.predict(xgb_input)[0])  # total kcal
    xgb_rate  = xgb_total / duration_hours              # kcal/h

    # --- LR hourly + total (DHS-based) ---
    d = str(demand).strip().capitalize()
    lr_input = pd.DataFrame([{
        "Weight": int(round(float(weight))),
        "CalorieDemand_Low": 1 if d == "Low" else 0,
        "CalorieDemand_Medium": 1 if d == "Medium" else 0  # High is baseline (both 0)
    }])
    lr_hourly = float(lr_model.predict(lr_input)[0])  # kcal/h
    lr_total  = lr_hourly * duration_hours            # total kcal

    # --- Blends ---
    rf_lr_rate   = rf_weight * rf_rate   + (1 - rf_weight) * lr_hourly
    xgb_lr_rate  = rf_weight * xgb_rate  + (1 - rf_weight) * lr_hourly
    rf_lr_total  = rf_lr_rate  * duration_hours
    xgb_lr_total = xgb_lr_rate * duration_hours

    return {
        "rf_only": round(rf_total, 2),
        "xgb_only": round(xgb_total, 2),
        "lr_only": round(lr_total, 2),
        "rf_lr": round(rf_lr_total, 2),
        "xgb_lr": round(xgb_lr_total, 2),
    }

HTML_FORM = """
<!doctype html>
<html><head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Calories Estimator</title>
  <style>
    body { font-family: system-ui,-apple-system,Arial,sans-serif; max-width:920px; margin:40px auto; padding:0 16px; }
    h1 { margin-bottom:8px }
    .card { background:#fafafa; border:1px solid #eee; border-radius:10px; padding:16px; margin-bottom:16px }
    form { display:grid; gap:10px; margin-top:16px }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:10px }
    input, select, button { padding:10px; font-size:16px }
    button { background:#111; color:#fff; border:0; border-radius:8px; cursor:pointer }
    button:hover { opacity:.9 }
    .result { font-size:18px; font-weight:600; margin-top:10px; }
    ul { line-height:1.8 }
  </style>
</head><body>
  <h1>Calories Estimator</h1>
  <div class="card">Enter your details and get calories burned predictions from all models.</div>

  <form method="post">
    <div class="row">
      <label>Gender
        <select name="gender">
          <option value="1" {{ m_sel }}>Male</option>
          <option value="0" {{ f_sel }}>Female</option>
        </select>
      </label>
      <label>Age (years)
        <input type="number" name="age" min="10" max="100" value="{{ age }}" required>
      </label>
    </div>

    <div class="row">
      <label>Height (cm)
        <input type="number" name="height" min="120" max="220" step="0.1" value="{{ height }}" required>
      </label>
      <label>Weight (kg)
        <input type="number" name="weight" min="30" max="200" step="0.1" value="{{ weight }}" required>
      </label>
    </div>

    <div class="row">
      <label>Duration (minutes)
        <input type="number" name="duration" min="1" max="300" step="1" value="{{ duration }}" required>
      </label>
      <label>Demand
        <select name="demand">
          <option {{ low }}>Low</option>
          <option {{ med }}>Medium</option>
          <option {{ high }}>High</option>
        </select>
      </label>
    </div>

    <button type="submit">Calculate</button>
  </form>

  {% if results %}
    <div class="card">
      <div class="result">Results (total kcal):</div>
      <ul>
        <li> Random Forest (only): <b>{{ results.rf_only }}</b> kcal</li>
        <li> XGBoost (only): <b>{{ results.xgb_only }}</b> kcal</li>
        <li> Linear Regression (only): <b>{{ results.lr_only }}</b> kcal</li>
        <li> RF × Linear Regression: <b>{{ results.rf_lr }}</b> kcal</li>
        <li> XGBoost × Linear Regression: <b>{{ results.xgb_lr }}</b> kcal</li>
      </ul>
    </div>
  {% endif %}
</body></html>
"""

@app.route("/", methods=["GET","POST"])
def index():
    ctx = dict(
        m_sel="selected", f_sel="",
        age="25", height="175", weight="70", duration="45",
        low="", med="selected", high="",
        results=None
    )

    if request.method == "POST":
        gender   = request.form.get("gender", "1")
        age      = request.form.get("age", "25")
        height   = request.form.get("height", "175")
        weight   = request.form.get("weight", "70")
        duration = request.form.get("duration", "45")
        demand   = request.form.get("demand", "Medium")

        results = predict_all(gender, age, height, weight, duration, demand)

        ctx.update({
            "m_sel": "selected" if gender == "1" else "",
            "f_sel": "selected" if gender == "0" else "",
            "age": age, "height": height, "weight": weight, "duration": duration,
            "low": "selected" if demand == "Low" else "",
            "med": "selected" if demand == "Medium" else "",
            "high": "selected" if demand == "High" else "",
            "results": results
        })

    return render_template_string(HTML_FORM, **ctx)

if __name__ == "__main__":
    app.run(debug=True)
