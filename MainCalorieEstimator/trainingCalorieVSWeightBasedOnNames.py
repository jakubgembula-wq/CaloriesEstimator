from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import re

INPUT_CSV = r"C:\Users\jakub\Desktop\CaloriesEstimator\MainCalorieEstimator\data\calories_burned_per_hour_wisconsin_dhs.csv"
SAVE_PATH = r"C:\Users\jakub\Desktop\CaloriesEstimator\MainCalorieEstimator\models\SecondaryCaloriesModelNameBased.joblib"

df = pd.read_csv(INPUT_CSV)

# Rename columns to standardized kg-based names when applicable.
rename_map = {
    "Calories_130_lbs": "Calories_59_kg",
    "Calories_155_lbs": "Calories_70_kg",
    "Calories_190_lbs": "Calories_86_kg"
}
df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

# Identify the available calorie columns.
cal_cols = [c for c in ["Calories_59_kg", "Calories_70_kg", "Calories_86_kg"] if c in df.columns]
if len(cal_cols) == 0:
    raise ValueError("No calories columns found. Expected 'Calories_59_kg', 'Calories_70_kg', 'Calories_86_kg'.")

# === Name-based intensity classifier ===
# Regular expression patterns grouped by intensity categories.
HIGH_PATTERNS = [
    r"\brun", r"\bracing\b", r"\bsprint", r"\bfootball\b", r"\bsoccer\b", r"\brugby\b",
    r"\bbasketball\b", r"\bice\s*hockey\b|\bhockey\b",
    r"\bbox(ing)?\b|\bkick(box| boxing)?\b",
    r"\bmartial\b|\bkarate\b|\btae\s*kwon\s*do\b|\bmuay\b|\bjudo\b",
    r"\bjump(ing)?\b|\brope\b", r"\bhiit\b|\binterval\b",
    r"\b(stair|step) ?(climb|mill|master)\b", r"\bspin(ning)?\b",
    r"\bski(ing)?\b|\bsnowboard(ing)?\b|\bxc\s*ski\b|\bdownhill\b",
    r"\brow(ing)?\b.*(vig|hard|race)", r"\bswim(ming)?\b.*(vig|butterfly|laps?)",
    r"\bmountain\b.*(bike|bicycl)", r"\bclimb(ing)?\b|\bbouldering\b",
    r"\baerobics.*high\b", r"\bcompetitive\b"
]

MEDIUM_PATTERNS = [
    r"\bcycle\b|\bbicycl(e|ing)\b|\bbike\b",
    r"\brow(ing)?\b", r"\bswim(ming)?\b",
    r"\btennis\b|\bpickleball\b|\bsquash\b|\bracquet(ball)?\b",
    r"\bskate\b|\broller(blade| skate)\b|\binline\b",
    r"\baerobics\b|\belliptical\b|\bcardio\b",
    r"\bdance\b|\bzumba\b",
    r"\byoga\b.*(power|vinyasa)", r"\bpilates\b",
    r"\bhiking\b|\bbackpack(ing)?\b",
    r"\b(stair|step)\b",
    r"\bshovel(ing)?\b|\bmowing\b|\byard\b",
    r"\bmoderate\b|\bmedium\b"
]

LOW_PATTERNS = [
    r"\bwalk(ing)?\b", r"\bstretch(ing)?\b", r"\byoga\b", r"\btai ?chi\b",
    r"\barchery\b", r"\bfishing\b", r"\bbowling\b", r"\bgolf\b",
    r"\bhousehold\b|\bclean(ing)?\b|\bdish(es|washing)\b|\blaundry\b|\bvacuum(ing)?\b",
    r"\bgardening\b|\bplant(ing)?\b|\bweeding\b",
    r"\bautomobile repair\b|\bcarpentry\b|\bhome repair\b",
    r"\barts? & crafts\b|\bcrafts?\b",
    r"\blight\b|\beasy\b|\bslow\b"
]

HIGH_RE = re.compile("|".join(HIGH_PATTERNS), flags=re.I)
MED_RE  = re.compile("|".join(MEDIUM_PATTERNS), flags=re.I)
LOW_RE  = re.compile("|".join(LOW_PATTERNS), flags=re.I)

# Classify an activity name into intensity level based on pattern matching.
def name_based_intensity(name: str) -> str:
    n = "" if pd.isna(name) else str(name)
    if HIGH_RE.search(n):
        return "High"
    if MED_RE.search(n):
        return "Medium"
    if LOW_RE.search(n):
        return "Low"
    # Heuristic fallback for terms related to sports or speed.
    if re.search(r"\b(game|match|sport|league)\b|\b\d+(\.\d+)?\s*(mph|km/h)\b|\bpac(e|ed)\b", n, flags=re.I):
        return "Medium"
    # Default classification.
    return "Low"

# Ensure that the expected 'Activity' column exists.
if "Activity" not in df.columns:
    raise ValueError("Expected an 'Activity' column with activity names.")

# Apply name-based classification.
df["CalorieDemand"] = df["Activity"].apply(name_based_intensity)

print("\nName-based classes (by Activity):")
print(df["CalorieDemand"].value_counts().to_string())

# Display formatting options.
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.colheader_justify", "center")

print(df.head())

# Convert dataset into long format for modeling.
long_df = df.melt(
    id_vars=["CalorieDemand"],
    value_vars=["Calories_59_kg", "Calories_70_kg", "Calories_86_kg"],
    var_name="WeightCol",
    value_name="Calories"
)

# Extract numeric weight values from column names.
long_df["Weight"] = long_df["WeightCol"].str.extract(r"(\d+)").astype(int)
long_df = long_df[["Weight", "CalorieDemand", "Calories"]]
print(long_df.head(10))

# One-hot encode categorical values and prepare training data.
X = pd.get_dummies(long_df[["Weight", "CalorieDemand"]], drop_first=True)
y = long_df["Calories"]

# Train linear regression model.
model = LinearRegression()
model.fit(X, y)

# Evaluate model performance on the training data.
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nR² (Accuracy): {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Saving the model (optional)
# import joblib, os
# os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
# joblib.dump(model, SAVE_PATH)
# print(f"\nModel saved successfully to:\n{SAVE_PATH}")
