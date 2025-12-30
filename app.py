import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------------------
# Load trained objects
# -----------------------------------------
model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("encoders.pkl", "rb"))
X_columns = pickle.load(open("columns.pkl", "rb"))

# Mapping for output
output_mapping = {0: "low", 1: "medium", 2: "high"}
st.set_page_config(page_title="Grocery Stock Predictor", page_icon="🛒", layout="wide")

st.markdown("""
<style>
/* Whole app background */
.stApp {
    background-color: #1E1E1E !important;
}

/* Remove white container */
.block-container {
    background-color: #1E1E1E !important;
}

/* Heading colors */
h1, h2, h3, label {
    color: #BB86FC !important;
    font-weight: bold;
}

/* Buttons */
.stButton>button {
    background-color: #8A2BE2 !important;
    color: white !important;
    font-size: 18px !important;
    border-radius: 10px !important;
    padding: 10px 25px !important;
}

/* Text input boxes */
input, textarea, select {
    background-color: #2C2C2C !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Set Dark UI Page Config + CSS
# -----------------------------------------
st.set_page_config(page_title="Grocery Stock Predictor", page_icon="🛒")

st.markdown("""
<style>
    body, .main {
        background-color: #1E1E1E;
    }
    h1, h2, h3, label, .stTextInput, .css-10trblm {
        color: #BB86FC !important;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #8A2BE2;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 25px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Safe transform
# -----------------------------------------
def safe_transform(le, value):
    if value not in le.classes_:
        le.classes_ = np.append(le.classes_, value)
    return le.transform([value])[0]

# Prediction
def predict(data_dict):
    row = pd.DataFrame([data_dict], columns=X_columns)
    for col in row.columns:
        if col in label_encoders:
            le = label_encoders[col]
            val = row[col].iloc[0]
            row[col] = safe_transform(le, val)
    pred_num = model.predict(row)[0]
    return output_mapping.get(pred_num, "unknown")

# -----------------------------------------
# UI Components
# -----------------------------------------
st.title("🛒 Grocery Stock Prediction System")
st.write("Enter grocery details to predict LOW / MEDIUM / HIGH stock levels.")

st.header("Enter Item Details")

item = st.text_input("Item Name", "Milk")
brand = st.text_input("Brand", "Amul")
store_type = st.selectbox("Store Type", ["kirana", "supermarket", "online"])
season = st.selectbox("Season", ["winter", "summer", "rainy", "monsoon"])
purchase_method = st.selectbox("Purchase Method", ["offline", "online"])

qty_bought = st.number_input("Quantity Bought", min_value=0, step=1)
days_used = st.number_input("Days Used", min_value=0, step=1)
daily_use = st.number_input("Daily Use (units/day)", min_value=0.0, step=0.1)
last_purchase_days_ago = st.number_input("Days Since Last Purchase", min_value=0, step=1)
household_size = st.number_input("Household Size", min_value=1, step=1)
price = st.number_input("Price", min_value=0.0)
discount_percent = st.number_input("Discount %", min_value=0, max_value=100, step=1)
shelf_life_days = st.number_input("Shelf Life Days", min_value=0, step=1)
rating = st.number_input("Rating", min_value=0.0, max_value=5.0, step=0.1)

# Predict button
if st.button("Predict Stock Level"):
    data = {
        "item": item,
        "brand": brand,
        "store_type": store_type,
        "season": season,
        "purchase_method": purchase_method,
        "qty_bought": qty_bought,
        "days_used": days_used,
        "daily_use": daily_use,
        "last_purchase_days_ago": last_purchase_days_ago,
        "household_size": household_size,
        "price": price,
        "discount_percent": discount_percent,
        "shelf_life_days": shelf_life_days,
        "rating": rating
    }

    result = predict(data)
    colors = {"low": "#E63946", "medium": "#F4A261", "high": "#2ECC71"}

    st.markdown(
        f"""
        <div style='background-color:{colors[result]};
                    padding:18px;
                    border-radius:12px;
                    margin-top:20px;
                    text-align:center;
                    font-size:30px;
                    color:black;
                    font-weight:bold;'
        >
            STOCK LEVEL: {result.upper()}
        </div>
        """,
        unsafe_allow_html=True
    )
