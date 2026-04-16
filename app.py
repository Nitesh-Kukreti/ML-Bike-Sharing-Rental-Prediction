# app.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Load model and encoder
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

# Load dataset for plotting & evaluation
train = pd.read_csv("dataSet/train.csv")
train['Season'] = le.transform(train['Season'])

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

# Create validation split (same random_state for consistency)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pred_val = model.predict(X_val)

while True:
    print("\nChoose an option:")
    print("1. Predict bikes for user input")
    print("2. Show graphs for individual conditions")
    print("3. Show Actual vs Predicted + Model Accuracy")
    print("4. Exit")

    choice = input("Enter choice: ")

    if choice == '1':
        print("\nEnter conditions to predict bike rentals:")

        temp = float(input("Temperature (°C): "))
        humidity = float(input("Humidity (%): "))
        wind = float(input("Wind Speed (km/h): "))
        rainfall = float(input("Rainfall (mm): "))
        snowfall = float(input("Snowfall (cm): "))
        hour = int(input("Hour (0-23): "))
        season_input = input("Season (Winter/Spring/Summer/Autumn): ").capitalize()
        is_holiday = int(input("Is Holiday? (0 = No, 1 = Yes): "))
        is_functioning = int(input("Is Functioning Day? (0 = No, 1 = Yes): "))

        try:
            season_encoded = le.transform([season_input])[0]
        except:
            print("Invalid season input! Defaulting to Winter.")
            season_encoded = le.transform(["Winter"])[0]

        user_features = pd.DataFrame([[
            temp, humidity, wind, rainfall, snowfall,
            hour, season_encoded, is_holiday, is_functioning
        ]], columns=features)

        predicted_bikes = model.predict(user_features)[0]
        print(f"\nPredicted Bikes Rented: {int(predicted_bikes)}")

    elif choice == '2':
        print("\nAvailable conditions to plot:")
        print(", ".join(features))

        feature_choice = input("Enter feature name exactly as shown: ").strip()

        if feature_choice not in features:
            print("Invalid feature!")
            continue

        plt.figure(figsize=(6,4))

        if feature_choice in ['Hour', 'IsHoliday', 'IsFunctioningDay', 'Season']:
            sns.barplot(x=feature_choice, y='Bikes_Rented', data=train)
        else:
            sns.scatterplot(x=feature_choice, y='Bikes_Rented', data=train, alpha=0.5)

        plt.title(f"Effect of {feature_choice} on Bikes Rented")
        plt.xlabel(feature_choice)
        plt.ylabel("Bikes Rented")
        plt.tight_layout()
        plt.show()

    elif choice == '3':
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        import numpy as np

        r2 = r2_score(y_val, pred_val)
        mae = mean_absolute_error(y_val, pred_val)
        mse = mean_squared_error(y_val, pred_val)
        rmse = np.sqrt(mse)

        print("\nModel Performance:")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Error: {mae:.2f} bikes")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f} bikes")

        # Plot scatter
        plt.figure(figsize=(6,6))
        sns.scatterplot(x=y_val, y=pred_val, alpha=0.5)
        plt.plot([0, max(y_val)], [0, max(y_val)], color='red', linestyle='--')
        plt.xlabel("Actual Bikes Rented")
        plt.ylabel("Predicted Bikes Rented")
        plt.title("Actual vs Predicted")
        plt.tight_layout()
        plt.show()

    elif choice == '4':
        print("Exiting program.")
        break

    else:
        print("Invalid choice.")
