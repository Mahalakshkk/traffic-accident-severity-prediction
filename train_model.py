import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("US_Accidents_March23.csv")
print("Dataset Loaded:", df.shape)

# Sample for faster training
df = df.sample(200000, random_state=42)

# =========================
# 2. Select Required Columns
# =========================
columns = [
    'Severity',
    'Weather_Condition',
    'Visibility(mi)',
    'Temperature(F)',
    'Wind_Speed(mph)',
    'Precipitation(in)',
    'Sunrise_Sunset'
]
df = df[columns]

# =========================
# 3. Handle Missing Values
# =========================
df = df.ffill()

# =========================
# 4. Target Adjustment
# =========================
df['Severity'] = df['Severity'] - 1

# =========================
# 5. Encode Categorical Features
# =========================
weather_encoder = LabelEncoder()
df['Weather_Condition'] = weather_encoder.fit_transform(df['Weather_Condition'])

df['Sunrise_Sunset'] = df['Sunrise_Sunset'].map({
    'Night': 0,
    'Day': 1
})

# =========================
# 6. Split Features & Target
# =========================
X = df.drop('Severity', axis=1)
y = df['Severity']

# =========================
# 7. Feature Scaling
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 8. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# 9. Train XGBoost Model
# =========================
model = XGBClassifier(
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train, y_train)

# =========================
# 10. Model Evaluation
# =========================
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# =========================
# 11. Feature Importance_toggle Importances Plot
# =========================
plt.figure(figsize=(8, 5))
sns.barplot(
    x=model.feature_importances_,
    y=X.columns
)
plt.title("Feature Importance - Accident Severity")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# =========================
# 12. Save Model & Preprocessors
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(weather_encoder, open("encoder.pkl", "wb"))

print("✅ model.pkl, scaler.pkl, encoder.pkl saved successfully")
print("✅ feature_importance.png saved successfully")

