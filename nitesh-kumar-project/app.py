import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------- LOAD DATA --------------------

df = pd.read_csv("dataset/train.csv")

# Encode Season
season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}
df["Season"] = df["Season"].map(season_map)

# Features & Target
X = df[[
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Hour",
    "Season",
    "IsHoliday",
    "IsFunctioningDay"
]]

y = df["Bikes_Rented"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predictions for Accuracy
y_pred = model.predict(X_test)

# -------------------- SIDEBAR --------------------

st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Select Feature",
    ["Predict Bikes", "View Graphs", "Model Accuracy"]
)

st.title("🚴 Bike Sharing Rental Prediction System")

# -------------------- FEATURE 1 --------------------

if option == "Predict Bikes":

    st.header("Bike Demand Prediction")

    temp = st.slider("Temperature (°C)", -20.0, 50.0, 10.0)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    wind = st.slider("Wind Speed (km/h)", 0.0, 50.0, 5.0)
    hour = st.slider("Hour of Day", 0, 23, 12)
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])
    holiday = st.selectbox("Is Holiday?", [0, 1])
    functioning = st.selectbox("Is Functioning Day?", [0, 1])

    season_encoded = season_map[season]

    if st.button("Predict Bikes Rented"):

        input_data = pd.DataFrame([[temp, humidity, wind, hour,
                                    season_encoded, holiday, functioning]],
                                  columns=X.columns)

        prediction = model.predict(input_data)[0]

        st.success(f"Predicted Bikes Rented: {int(prediction)} 🚲")

# -------------------- FEATURE 2 --------------------

elif option == "View Graphs":

    st.header("Condition Analysis Graphs")

    graph_option = st.selectbox(
        "Select Condition",
        ["Temperature", "Humidity", "Wind_Speed", "Hour"]
    )

    fig, ax = plt.subplots()
    ax.scatter(df[graph_option], df["Bikes_Rented"])
    ax.set_xlabel(graph_option)
    ax.set_ylabel("Bikes Rented")
    ax.set_title(f"{graph_option} vs Bikes Rented")

    st.pyplot(fig)

# -------------------- FEATURE 3 --------------------

elif option == "Model Accuracy":

    st.header("Model Performance")

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.write(f"✅ R² Score: {r2:.3f}")
    st.write(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"✅ Mean Squared Error (MSE): {mse:.2f}")

    # -------- Accuracy Graph --------

    st.subheader("Actual vs Predicted Bikes Rented")

    fig, ax = plt.subplots()

    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Bikes Rented")
    ax.set_ylabel("Predicted Bikes Rented")
    ax.set_title("Actual vs Predicted")
    ax.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()])


    st.pyplot(fig)

