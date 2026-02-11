# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Load dataset
train = pd.read_csv("dataSet/train.csv")

# Encode Season
le = LabelEncoder()
train['Season'] = le.fit_transform(train['Season'])

# Features
features = [
    'Temperature',
    'Humidity',
    'Wind_Speed',
    'Rainfall',
    'Snowfall',
    'Hour',
    'Season',
    'IsHoliday',
    'IsFunctioningDay'
]

X = train[features]
y = train['Bikes_Rented']

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
pred_val = model.predict(X_val)

r2 = r2_score(y_val, pred_val)
mae = mean_absolute_error(y_val, pred_val)

print("Model Performance:")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.2f} bikes")

# Save model and encoder
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nModel and encoder saved successfully.")
