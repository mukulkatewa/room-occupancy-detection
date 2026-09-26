"""Streamlit demo for the room occupancy detection model."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.train import FEATURES, build_candidates, load

MODEL_PATH = Path("models/model.joblib")
DATA_DIR = Path("data")


@st.cache_resource
def get_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    train_df = load(DATA_DIR / "datatraining.csv")
    model = build_candidates()["logistic_regression"]
    model.fit(train_df[FEATURES], train_df["Occupancy"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


st.set_page_config(page_title="Room Occupancy Detection", page_icon="🚪")
st.title("🚪 Room Occupancy Detection")
st.caption(
    "Predicts whether a room is occupied from ambient sensor readings. "
    "Logistic Regression classifier, holdout accuracy = 98.5%."
)

model = get_model()

# Median sensor readings for each class in the training data — used as
# realistic one-click scenarios rather than arbitrary numbers.
SCENARIOS = {
    "Custom": None,
    "Empty room, lights off (typical unoccupied reading)": {
        "Temperature": 20.20, "Humidity": 26.19, "Light": 0.0, "CO2": 446.0, "HumidityRatio": 0.00378,
    },
    "Meeting in progress (typical occupied reading)": {
        "Temperature": 21.77, "Humidity": 26.44, "Light": 454.0, "CO2": 944.0, "HumidityRatio": 0.00435,
    },
    "Empty room, lights left on (edge case)": {
        "Temperature": 20.20, "Humidity": 26.19, "Light": 400.0, "CO2": 450.0, "HumidityRatio": 0.00378,
    },
}

DEFAULTS = {"Temperature": 23.2, "Humidity": 27.3, "Light": 450.0, "CO2": 720.0, "HumidityRatio": 0.0048}
KEYS = {"Temperature": "temp_in", "Humidity": "humidity_in", "Light": "light_in", "CO2": "co2_in", "HumidityRatio": "hr_in"}
for field, key in KEYS.items():
    st.session_state.setdefault(key, DEFAULTS[field])


def apply_scenario():
    preset = SCENARIOS[st.session_state["scenario_choice"]]
    if preset:
        for field, key in KEYS.items():
            st.session_state[key] = preset[field]


st.selectbox(
    "Try a scenario (from real sensor data)",
    list(SCENARIOS.keys()),
    key="scenario_choice",
    on_change=apply_scenario,
)

col1, col2 = st.columns(2)
with col1:
    temperature = st.number_input("Temperature (°C)", step=0.1, key=KEYS["Temperature"])
    humidity = st.number_input("Humidity (%)", step=0.1, key=KEYS["Humidity"])
    light = st.number_input("Light (lux)", step=1.0, key=KEYS["Light"])
with col2:
    co2 = st.number_input("CO2 (ppm)", step=1.0, key=KEYS["CO2"])
    humidity_ratio = st.number_input("Humidity ratio", step=0.0001, format="%.4f", key=KEYS["HumidityRatio"])

if st.button("Predict occupancy", type="primary"):
    row = pd.DataFrame(
        [
            {
                "Temperature": temperature,
                "Humidity": humidity,
                "Light": light,
                "CO2": co2,
                "HumidityRatio": humidity_ratio,
            }
        ]
    )
    prediction = int(model.predict(row)[0])
    if prediction == 1:
        st.success("Prediction: Occupied ✅")
    else:
        st.info("Prediction: Not occupied")

st.divider()
st.caption("Source: [GitHub repo](https://github.com/mukulkatewa/room-occupancy-detection)")
