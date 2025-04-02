import streamlit as st
import pandas as pd
import requests
import joblib
import time
import plotly.express as px


# Load trained AI model
def load_model():
    return joblib.load("predictive_maintenance_model.pkl")


# Fetch real-time data from ThingSpeak
def fetch_data():
    CHANNEL_ID = "2890659"  # Replace with your ThingSpeak Channel ID
    READ_API_KEY = "44SV31POUHPOVJ0M"  # Replace with your Read API Key
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

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


# Predict maintenance status based on sensor values
def predict_maintenance(sensor, value):
    # Define thresholds
    thresholds = {
        "Temperature": (20, 45),
        "Humidity": (25, 80),
        "Hall Effect": (0, 0),
        "Sound": (0, 95),
        "Current": (0, 7),
        "Gas": (0, 500),
        "Flame": (0, 0),
        "Relay Status": (0, 0)
    }

    min_val, max_val = thresholds.get(sensor, (0, 100))

    # Check if value is within the range
    if not (min_val <= value <= max_val):
        return f"⚠️ {sensor} Out of Range - Maintenance Required"

    return "🟢 Good Condition - No Maintenance Needed"


# Streamlit UI
st.set_page_config(page_title="IoT Predictive Maintenance", layout="wide")
st.title("IoT Predictive Maintenance Dashboard")

# Load AI Model
model = load_model()

# Fetch real-time data
sensor_data = fetch_data()

if sensor_data:
    # Create tabs for each sensor
    tabs = st.tabs(list(sensor_data.keys()))

    for i, sensor in enumerate(sensor_data.keys()):
        with tabs[i]:
            st.subheader(f"{sensor} Sensor")
            st.write(f"**Current Value:** {sensor_data[sensor]}")

            # Show sensor value visualization
            fig = px.bar(x=[sensor], y=[sensor_data[sensor]], labels={"x": "Sensor", "y": "Value"},
                         title=f"{sensor} Value", height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Predict maintenance status for this sensor
            prediction_result = predict_maintenance(sensor, sensor_data[sensor])
            st.markdown(
                f"<h2 style='color: {'red' if '⚠️' in prediction_result else 'green'}'>{prediction_result}</h2>",
                unsafe_allow_html=True)

    # Auto-refresh every 15 seconds
    time.sleep(15)
    st.rerun()
else:
    st.error("❌ Unable to fetch real-time sensor data. Please check your ThingSpeak channel.")
