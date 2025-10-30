import streamlit as st
import pandas as pd
import os
import joblib
import gzip
import shutil
import lzma
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------
# 🔹 Compress .pkl file to stay under 25 MB
# -----------------------------------------------------


# -----------------------------------------------
# 🔹 Compress .pkl manually with gzip
# -----------------------------------------------
def compress_pickle(input_path, target_size_mb=25):
    """Compress a pickle file to stay under target size."""
    temp_path = input_path + ".gz"

    with open(input_path, "rb") as f_in:
        with gzip.open(temp_path, "wb", compresslevel=8) as f_out:
            shutil.copyfileobj(f_in, f_out)

    compressed_size = os.path.getsize(temp_path) / (1024 * 1024)
    
    if compressed_size <= target_size_mb:
        os.remove(input_path)
        os.rename(temp_path, input_path)
    else:
        os.remove(temp_path)

# -----------------------------------------------------
# 🔹 Train model if not found
# -----------------------------------------------------
def train_and_save_model():
    df = pd.read_csv("Maharashtra_crop_dataset.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    X = df[[
        "season", "district", "soiltype", "avgrainfall_mm", "avgtemp_c", "avghumidity_%",
        "soil_ph", "nitrogen_kg_ha", "phosphorus_kg_ha", "potassium_kg_ha"
    ]]
    y = df["Crop"]

    X = pd.get_dummies(X, columns=["district", "soiltype", "season"], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "crop_recommendation.pkl", compress=8)
    joblib.dump(X.columns.tolist(), "model_columns.pkl", compress=7)

    compress_pickle("crop_recommendation.pkl", target_size_mb=25) # type: ignore

    return model, X.columns.tolist(), df

# -----------------------------------------------------
# 🔹 Load model and columns
# -----------------------------------------------------
@st.cache_resource
def load_model_and_columns():
    if not os.path.exists("crop_recommendation.pkl") or not os.path.exists("model_columns.pkl"):
        model, model_columns, df = train_and_save_model()
    else:
        model = joblib.load("crop_recommendation.pkl")
        model_columns = joblib.load("model_columns.pkl")
        df = pd.read_csv("Maharashtra_crop_dataset.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    return model, model_columns, df

model, model_columns, df = load_model_and_columns()

# -----------------------------------------------------
# 🌾 Streamlit UI
# -----------------------------------------------------
st.title("AI Powered Maharashtra Crop Recommendation System")
st.write("Enter your soil and weather conditions below to get top crop recommendations with expected yield and alerts.")

# Input section
available_districts = sorted(df["district"].unique())
available_soiltypes = sorted(df["soiltype"].unique())
available_seasons = sorted(df["season"].unique())

with st.form("crop_form"):
    st.subheader("Enter Farm Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        district = st.selectbox("District", available_districts)
        soiltype = st.selectbox("Soil Type", available_soiltypes)
        season = st.selectbox("Season", available_seasons)

    with col2:
        avgrainfall_mm = st.number_input("Average Rainfall (mm)", min_value=0.0, step=1.0)
        avgtemp_c = st.number_input("Average Temperature (°C)", min_value=0.0, step=0.1)
        avghumidity = st.number_input("Average Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)

    with col3:
        soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)
        nitrogen = st.number_input("Nitrogen (kg/ha)", min_value=0.0, step=1.0)
        phosphorus = st.number_input("Phosphorus (kg/ha)", min_value=0.0, step=1.0)
        potassium = st.number_input("Potassium (kg/ha)", min_value=0.0, step=1.0)

    submitted = st.form_submit_button("🔍 Get Crop Recommendations ")

# -----------------------------------------------------
# 🔹 Helper functions
# -----------------------------------------------------
def get_weather_alert(temp, humidity, rainfall):
    alerts = []
    if temp > 35:
        alerts.append("🌡️ High temperature — heat stress risk.")
    elif temp < 15:
        alerts.append("❄️ Low temperature — slow growth expected.")
    if humidity > 85:
        alerts.append("💧 High humidity — possible fungal risk.")
    elif humidity < 30:
        alerts.append("🔥 Low humidity — frequent irrigation needed.")
    if rainfall > 1200:
        alerts.append("☔ Heavy rainfall — ensure good drainage.")
    elif rainfall < 400:
        alerts.append("🌤️ Low rainfall — use drought-tolerant crops.")
    if not alerts:
        alerts.append("✅ Weather looks favorable for most crops.")
    return alerts

def get_soil_recommendation(ph):
    if ph < 6:
        return "Add lime to reduce soil acidity and improve nutrient uptake."
    elif ph > 8:
        return "Add organic matter or gypsum to balance alkaline soil."
    else:
        return "Soil pH is ideal — maintain organic content."

yield_profit_data = {
    "Cotton": ("18–25 quintals/hectare", "₹45,000–55,000"),
    "Soybean": ("20–30 quintals/hectare", "₹35,000–40,000"),
    "Tur": ("10–15 quintals/hectare", "₹25,000–30,000"),
    "Wheat": ("35–45 quintals/hectare", "₹30,000–40,000"),
    "Jowar": ("20–30 quintals/hectare", "₹20,000–30,000"),
    "Rice": ("40–55 quintals/hectare", "₹35,000–40,000"),
    "Gram": ("12–18 quintals/hectare", "₹28,000–35,000"),
    "Sugarcane": ("800–1000 quintals/hectare", "₹80,000–1,00,000"),
    "Maize": ("40–50 quintals/hectare", "₹30,000–40,000"),
    "Groundnut": ("20–25 quintals/hectare", "₹35,000–45,000"),
}

# ----------------------
# 🔹 Prediction logic (fixed scaling and visible cards)
# ----------------------
if submitted:
    try:
        # Prepare input data
        user_data = pd.DataFrame([{
            "district": district, "soiltype": soiltype, "season": season,
            "avgrainfall_mm": avgrainfall_mm, "avgtemp_c": avgtemp_c,
            "avghumidity_%": avghumidity, "soil_ph": soil_ph,
            "nitrogen_kg_ha": nitrogen, "phosphorus_kg_ha": phosphorus, "potassium_kg_ha": potassium
        }])
        user_data = pd.get_dummies(user_data, columns=["district", "soiltype", "season"], drop_first=True)
        user_data = user_data.reindex(columns=model_columns, fill_value=0)

        # Predict probabilities for all crops
        probs = model.predict_proba(user_data)[0]
        crops = model.classes_

        crop_probs = sorted(list(zip(crops, probs)), key=lambda x: x[1], reverse=True)
        top3 = crop_probs[:3]

# get relative differences but exaggerate scaling
        p_max = max(p for _, p in top3)
        scaled_list = []
        for i, (crop, prob) in enumerate(top3):
        # make sure even low probs get high % range
            relative_score = (prob / p_max) if p_max > 0 else 0
        scaled_percent = 85 + (relative_score * 10)  # 85–95%
        scaled_list.append((crop, round(min(95, max(85, scaled_percent)), 1)))


        # Extract the raw probs for top3
        raw_vals = [p for (_, p) in top3]
        p_max = max(raw_vals)
        p_min = min(raw_vals)

        # Robust mapping of raw probabilities into the 90 -> 80 range
        # If all probs equal (p_max == p_min), assign fixed [90,85,80]
        scaled_list = []
        if p_max == p_min:
            fixed = [90.0, 85.0, 80.0]
            for (cp, _), s in zip(top3, fixed):
                scaled_list.append((cp, s))
        else:
            for crop_name, prob in top3:
                # linear interpolation: map prob in [p_min, p_max] -> [80, 90]
                scaled = 80.0 + ((prob - p_min) / (p_max - p_min)) * 10.0
                # clamp just in case
                scaled = max(80.0, min(95.0, scaled))
                scaled_list.append((crop_name, scaled))

        # Display Results (visible cards, light background)
        st.subheader("🌾 Prediction Results ")
        for i, (crop, match_percent) in enumerate(scaled_list):
            yield_est, profit_est = yield_profit_data.get(crop, ("N/A", "N/A"))
            # Use pleasant light backgrounds so text is visible
            card_bg = "#1c1919" if i % 2 == 0 else "#161614"
            st.markdown(f"""
                <div style="
                    background-color:{card_bg};
                    border-radius:10px;
                    padding:14px;
                    margin-bottom:12px;
                    border-left:6px solid #10b981;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
                    color: #0f172a;
                ">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div style="font-size:18px;font-weight:600;">🌱 {crop}</div>
                        <div style="font-size:16px;color:#065f46;font-weight:700;">{match_percent:.1f}% Match</div>
                    </div>
                    <div style="margin-top:8px;font-size:14px;color:#0f172a;">
                        🌾 <b>Expected Yield:</b> {yield_est} &nbsp;&nbsp; | &nbsp;&nbsp; 💰 <b>Estimated Profit:</b> {profit_est}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Weather Alerts
        st.markdown("### 🌦️ Weather Alert")
        for alert in get_weather_alert(avgtemp_c, avghumidity, avgrainfall_mm):
            st.info(alert)

        # Soil Recommendation
        st.markdown("### 🌱 Soil Recommendation")
        st.warning(get_soil_recommendation(soil_ph))

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
