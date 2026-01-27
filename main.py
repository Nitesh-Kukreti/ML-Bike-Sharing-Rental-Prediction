# main.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 1. Load dataset
# -------------------------------
train = pd.read_csv("dataSet/train.csv")

# Encode categorical feature 'Season'
le = LabelEncoder()
train['Season'] = le.fit_transform(train['Season'])

# Features and target
features = ['Temperature', 'Humidity', 'Wind_Speed', 'Hour', 'Season', 'IsHoliday', 'IsFunctioningDay']
X = train[features]
y = train['Bikes_Rented']

# -------------------------------
# 2. Train/validation split
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validation predictions for Actual vs Predicted
pred_val = model.predict(X_val)

# -------------------------------
# 3. Menu loop
# -------------------------------
while True:
    print("\nChoose an option:")
    print("1. Predict bikes for user input")
    print("2. Show graphs for individual conditions")
    print("3. Show Actual vs Predicted scatter plot")
    print("n. Exit")
    choice = input("Enter choice: ").lower()

    if choice == '1':
        # --- User input prediction ---
        print("\nEnter conditions to predict bike rentals:")
        temp = float(input("Temperature (°C): "))
        humidity = float(input("Humidity (%): "))
        wind = float(input("Wind Speed (km/h): "))
        hour = int(input("Hour (0-23): "))
        season_input = input("Season (Winter/Spring/Summer/Autumn): ").capitalize()
        is_holiday = int(input("Is Holiday? (0 = No, 1 = Yes): "))
        is_functioning = int(input("Is Functioning Day? (0 = No, 1 = Yes): "))

        # Encode season
        try:
            season_encoded = le.transform([season_input])[0]
        except:
            print("Invalid season input! Defaulting to Winter.")
            season_encoded = le.transform(["Winter"])[0]

        # Prepare DataFrame
        user_features = pd.DataFrame([[temp, humidity, wind, hour, season_encoded, is_holiday, is_functioning]],
                                     columns=features)

        # Predict
        predicted_bikes = model.predict(user_features)[0]
        print(f"\nPredicted Bikes Rented: {int(predicted_bikes)}")

    elif choice == '2':
        # --- Individual feature graphs ---
        print("\nAvailable conditions to plot: Temperature, Hour, Humidity, Wind_Speed, Season, IsHoliday, IsFunctioningDay")
        feature_choice = input("Enter feature name: ").strip()
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
        plt.show()

    elif choice == '3':
        # --- Actual vs Predicted ---
        plt.figure(figsize=(6,6))
        sns.scatterplot(x=y_val, y=pred_val, alpha=0.5)
        plt.plot([0, max(y_val)], [0, max(y_val)], color='red', linestyle='--')
        plt.xlabel("Actual Bikes Rented")
        plt.ylabel("Predicted Bikes Rented")
        plt.title("Actual vs Predicted")
        plt.show()

    elif choice == 'n':
        print("Exiting program. Goodbye!")
        break

    else:
        print("Invalid choice. Please try again.")
