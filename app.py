import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# -----------------------------
# LOAD MODELS
# -----------------------------
rf_model = pickle.load(open("final_rf_model.pkl", "rb"))
clf_model = pickle.load(open("price_category_model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# -----------------------------
# BRAND IMAGE FUNCTION
# -----------------------------
def show_brand_image(brand):
    path = f"images/{brand.lower()}.png"
    if os.path.exists(path):
        st.image(path, width=120)

# -----------------------------
# TITLE
# -----------------------------
st.title("📱 Smartphone Price & Category Prediction")

import os

image_path = os.path.join("assets", "app_preview.png")
st.image(image_path, use_container_width=True)

st.write("Enter smartphone specifications to predict price and category")

# -----------------------------
# INPUTS
# -----------------------------
ram = st.slider("RAM (GB)", 2, 24, 8)
storage = st.slider("Storage (GB)", 32, 512, 128)
battery = st.slider("Battery (mAh)", 2000, 8000, 5000)
fast_charge = st.slider("Fast Charging (W)", 10, 150, 30)
rear_cam = st.slider("Rear Camera (MP)", 8, 200, 50)
front_cam = st.slider("Front Camera (MP)", 5, 50, 16)
wifi = st.selectbox("WiFi Version", [5, 6, 6.5])

brand = st.selectbox("Brand", [
    "Apple","Samsung","Xiaomi","OnePlus",
    "Oppo","Vivo","Realme","Google","Motorola","Nothing"
])

show_brand_image(brand)

# -----------------------------
# BUILD INPUT DATA
# -----------------------------
input_dict = {col: 0 for col in features}

# numeric features
input_dict['ram_gb'] = ram
input_dict['storage_gb'] = storage
input_dict['battery_mah'] = battery
input_dict['fast_charging_w'] = fast_charge
input_dict['rear_camera_mp'] = rear_cam
input_dict['front_camera_mp'] = front_cam
input_dict['wifi_version'] = wifi

# brand one-hot encoding
brand_col = f"brand_{brand}"
if brand_col in input_dict:
    input_dict[brand_col] = 1

# convert to dataframe
input_df = pd.DataFrame([input_dict])

# -----------------------------
# PREDICT
# -----------------------------
if st.button("Predict"):
    price = rf_model.predict(input_df)[0]
    category = clf_model.predict(input_df)[0]

    st.success(f"💰 Predicted Price: ₹{int(price)}")
    st.info(f"🏷️ Category: {category}")