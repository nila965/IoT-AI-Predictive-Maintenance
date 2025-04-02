# app.py
import streamlit as st
import pandas as pd
import requests
import joblib
import time
import plotly.express as px

# ThingSpeak Configuration
CHANNEL_ID = "2890659"  # 🔹 Your ThingSpeak Channel ID
READ_API_KEY = "44SV31POUHPOVJ0M"  # 🔹 Your ThingSpeak Read API Key
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"


# Load trained model
def load_model():
    return joblib.load("predictive_maintenance_model.pkl")


# Safe Thresholds for Good Condition
SAFE_RANGES = {
    "Temperature": (20, 40),
    "Humidity": (30, 80),
    "Hall Effect": (0, 1),
    "Sound": (20, 200),
    "Current": (0, 500),
    "Gas": (0, 500),
    "Flame": (0, 1),
    "Relay Status": (0, 1),
}


# Fetch real-time data from ThingSpeak
def fetch_data():
    try:
        response = requests.get(THINGSPEAK_URL)
        response.raise_for_status()  # Raise an error for HTTP issues
        data = response.json()  # Parse JSON

        if "feeds" not in data or not data["feeds"]:
            st.warning("⚠️ No real-time data available from ThingSpeak")
            return None

        feeds = data["feeds"][0]

        return {
            "Temperature": float(feeds.get("field1", 0)),
            "Humidity": float(feeds.get("field2", 0)),
            "Hall Effect": int(feeds.get("field3", 0)),
            "Sound": float(feeds.get("field4", 0)),
            "Current": float(feeds.get("field5", 0)),
            "Gas": float(feeds.get("field6", 0)),
            "Flame": int(feeds.get("field7", 0)),
            "Relay Status": int(feeds.get("field8", 0))
        }

    except requests.exceptions.RequestException as e:
        st.error(f"🌐 ThingSpeak Connection Error: {e}")
        return None


# Predict maintenance status
def predict_maintenance(data, model):
    # Check if sensor values are within safe limits
    safe_condition = all(SAFE_RANGES[key][0] <= data[key] <= SAFE_RANGES[key][1] for key in SAFE_RANGES)

    if safe_condition:
        return "🟢 Good Condition"

    # Else, use AI Model for prediction
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return "🔴 Maintenance Required" if prediction == 1 else "🟢 Good Condition"


# Streamlit App UI
st.set_page_config(page_title="IoT Predictive Maintenance", layout="wide")
st.title("IoT Predictive Maintenance")
st.sidebar.header("Sensor Data")

# Load AI Model
model = load_model()

# Fetch real-time data
sensor_data = fetch_data()

if sensor_data:
    # Sidebar Sensor Inputs
    selected_sensor = st.sidebar.selectbox("Select Sensor", list(sensor_data.keys()))

    # Display Sensor Data
    st.sidebar.write(f"**{selected_sensor} Value:** {sensor_data[selected_sensor]}")

    # Data Visualization
    st.subheader(f"{selected_sensor} Data Visualization")
    fig = px.bar(x=[selected_sensor], y=[sensor_data[selected_sensor]], labels={"x": "Sensor", "y": "Value"},
                 title=f"{selected_sensor} Value", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # AI Prediction
    st.subheader("AI Prediction for Maintenance")
    prediction_result = predict_maintenance(sensor_data, model)
    st.markdown(f"<h2 style='color: {'red' if '🔴' in prediction_result else 'green'}'>{prediction_result}</h2>",
                unsafe_allow_html=True)

    # Auto-refresh every 15 seconds
    time.sleep(15)
    st.experimental_rerun()
else:
    st.error("❌ Unable to fetch real-time sensor data. Please check your ThingSpeak channel.")
