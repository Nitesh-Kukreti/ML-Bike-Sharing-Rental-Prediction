import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Bike Rental Prediction", layout="wide")

# -----------------------------
# Load Model & Encoder
# -----------------------------
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

# Load Dataset
data = pd.read_csv("dataSet/train.csv")
data['Season'] = le.transform(data['Season'])

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

X = data[features]
y = data['Bikes_Rented']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pred_val = model.predict(X_val)

# -----------------------------
# Header
# -----------------------------
st.title("🚲 Bike Rental Demand Prediction System")
st.markdown("Predict daily bike rentals based on weather and time conditions.")
st.caption("Developed by Nitesh | CSE | Roll No: 2323046")

st.divider()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Input Conditions")

st.sidebar.subheader("Weather")
temp = st.sidebar.number_input("Temperature (°C)", value=20.0)
humidity = st.sidebar.number_input("Humidity (%)", value=50.0)
wind = st.sidebar.number_input("Wind Speed (km/h)", value=5.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", value=0.0)
snowfall = st.sidebar.number_input("Snowfall (cm)", value=0.0)

st.sidebar.subheader("Time")
hour = st.sidebar.slider("Hour", 0, 23, 12)
season_input = st.sidebar.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])

st.sidebar.subheader("Day Type")
is_holiday = st.sidebar.selectbox("Is Holiday?", [1, 0])
is_functioning = st.sidebar.selectbox("Is Functioning Day?", [1, 0])

# -----------------------------
# Prediction Section
# -----------------------------
st.header("🔮 Prediction")

if st.button("Predict Rentals"):

    season_encoded = le.transform([season_input])[0]

    user_input = pd.DataFrame([[
        temp, humidity, wind, rainfall, snowfall,
        hour, season_encoded, is_holiday, is_functioning
    ]], columns=features)

    prediction = int(model.predict(user_input)[0])

    col1, col2 = st.columns(2)

    col1.metric("Predicted Rentals", prediction)

    # Demand Level
    if prediction < 300:
        level = "Low Demand"
    elif prediction < 700:
        level = "Medium Demand"
    else:
        level = "High Demand"

    col2.metric("Demand Level", level)

st.divider()

# -----------------------------
# Condition Analysis
# -----------------------------
st.header("📊 Condition Analysis")

selected_feature = st.selectbox("Select Feature to Analyze", features)

if selected_feature:

    if selected_feature in ['Hour', 'IsHoliday', 'IsFunctioningDay', 'Season']:
        fig = px.bar(
            data,
            x=selected_feature,
            y='Bikes_Rented',
            title=f"{selected_feature} vs Bikes Rented"
        )
    else:
        fig = px.scatter(
            data,
            x=selected_feature,
            y='Bikes_Rented',
            opacity=0.5,
            title=f"{selected_feature} vs Bikes Rented"
        )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------
# Model Performance
# -----------------------------
st.header("📈 Model Performance")

r2 = r2_score(y_val, pred_val)
mae = mean_absolute_error(y_val, pred_val)

col1, col2 = st.columns(2)
col1.metric("R² Score", f"{r2:.4f}")
col2.metric("MAE (Bikes)", f"{mae:.2f}")

# Actual vs Predicted Plot
fig_perf = go.Figure()

fig_perf.add_trace(go.Scatter(
    x=y_val,
    y=pred_val,
    mode='markers',
    name='Predicted',
    opacity=0.5
))

fig_perf.add_trace(go.Scatter(
    x=[0, max(y_val)],
    y=[0, max(y_val)],
    mode='lines',
    name='Perfect Prediction'
))

fig_perf.update_layout(
    title="Actual vs Predicted Rentals",
    xaxis_title="Actual Bikes Rented",
    yaxis_title="Predicted Bikes Rented"
)

st.plotly_chart(fig_perf, use_container_width=True)
