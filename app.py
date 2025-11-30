# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import xgboost as xgb

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="California Home Price Predictor",
    layout="wide"
)

# ------------------------------------------------------------
# Load model + feature order
# ------------------------------------------------------------
# Load XGBoost JSON model (safe for all versions)
booster = xgb.Booster()
booster.load_model("xgb_model.json")

# Load feature order for prediction
with open("features_nolist.json", "r") as f:
    feature_order = json.load(f)

# ------------------------------------------------------------
# Model metrics (static — from your notes)
# ------------------------------------------------------------
R2 = 0.89
MAPE = 10.54
MdAPE = 7.07
TRAIN_SAMPLES = 44142

# Pretty names for visualization
pretty_names = {
    "DaysOnMarket": "Days on Market",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "BathroomsTotalInteger": "Bathrooms",
    "LivingArea": "Living Area (sqft)",
    "FireplaceYN": "Fireplace",
    "YearBuilt": "Year Built",
    "ParkingTotal": "Parking Spaces",
    "BedroomsTotal": "Bedrooms",
    "PoolPrivateYN": "Private Pool",
    "LotSizeAcres": "Lot Size (acres)",
    "Stories": "Stories"
}

# ------------------------------------------------------------
# App Title
# ------------------------------------------------------------
st.title("California Home Price Prediction")

# Tabs
tab_pred, tab_importance, tab_summary = st.tabs(
    ["Prediction", "Feature Importance", "Model Summary"]
)

# ------------------------------------------------------------
# Tab 1: Prediction
# ------------------------------------------------------------
with tab_pred:
    st.subheader("Enter Property Details")

    with st.form("prediction_form"):

        # LOCATION
        st.markdown("#### Location")
        c1, c2 = st.columns(2)

        with c1:
            Latitude = st.number_input(
                "Latitude", 32.0, 42.0, step=0.001
            )
        with c2:
            Longitude = st.number_input(
                "Longitude", -124.0, -114.0, step=0.001
            )

        # SIZE
        st.markdown("#### Size")
        c3, c4 = st.columns(2)
        with c3:
            LivingArea = st.number_input("Living Area (sqft)", 1.0, step=50.0)
        with c4:
            LotSizeAcres = st.number_input("Lot Size (acres)", 0.0, step=0.01)

        Stories = st.number_input("Stories", 1.0, step=0.5)

        # BEDS & BATHS
        st.markdown("#### Bedrooms & Bathrooms")
        c5, c6 = st.columns(2)

        with c5:
            BedroomsTotal = st.number_input("Bedrooms", 1)
        with c6:
            BathroomsTotalInteger = st.number_input("Bathrooms", 0.5, step=0.5)

        # FEATURES
        st.markdown("#### Home Features")
        c7, c8 = st.columns(2)

        with c7:
            FireplaceYN = st.selectbox("Fireplace", ["No", "Yes"])
        with c8:
            PoolPrivateYN = st.selectbox("Private Pool", ["No", "Yes"])

        ParkingTotal = st.number_input("Total Parking", 0, step=1)

        # MARKET
        st.markdown("#### Market Info")
        c9, c10 = st.columns(2)

        with c9:
            DaysOnMarket = st.number_input("Days on Market", 0, step=1)
        with c10:
            YearBuilt = st.number_input("Year Built", 1800, pd.Timestamp.now().year)

        submitted = st.form_submit_button("Predict Price")

    # Validation
    def validate():
        errors = []
        if BathroomsTotalInteger > BedroomsTotal + 3:
            errors.append("Bathrooms unusually high relative to bedrooms.")
        if LotSizeAcres > 0 and LivingArea > LotSizeAcres * 43560:
            errors.append("Living area exceeds total lot size.")
        return errors

    if submitted:
        errors = validate()
        if errors:
            st.error("Fix the following issues:")
            for e in errors:
                st.write("- " + e)
            st.stop()

        bool_map = {"No": 0, "Yes": 1}
        FireplaceYN = bool_map[FireplaceYN]
        PoolPrivateYN = bool_map[PoolPrivateYN]

        # build row in the correct order
        row = pd.DataFrame([[
            DaysOnMarket,
            Latitude,
            Longitude,
            BathroomsTotalInteger,
            LivingArea,
            FireplaceYN,
            YearBuilt,
            ParkingTotal,
            BedroomsTotal,
            PoolPrivateYN,
            LotSizeAcres,
            Stories
        ]], columns=feature_order)

        # Convert to DMatrix (stable for XGBoost JSON models)
        dmatrix = xgb.DMatrix(row)

        # Predict log-price then exponentiate
        y_log = booster.predict(dmatrix)[0]
        y_pred = np.expm1(y_log)

        # Confidence band using MdAPE
        lower = y_pred / (1 + MdAPE/100)
        upper = y_pred * (1 + MdAPE/100)

        st.success(f"### Estimated Close Price: **${y_pred:,.0f}**")
        st.write(f"**Confidence Range:** ${lower:,.0f} → ${upper:,.0f}")

# ------------------------------------------------------------
# Tab 2: Feature Importance
# ------------------------------------------------------------
with tab_importance:
    st.subheader("Feature Importance (XGBoost)")

    importances = booster.get_score(importance_type="weight")

    # convert key names to pretty form
    items = []
    for feat, score in importances.items():
        name = pretty_names.get(feat, feat)
        items.append((name, score))

    items = sorted(items, key=lambda x: x[1])

    labels = [i[0] for i in items]
    values = [i[1] for i in items]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, values, color="#4B9CD3")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# ------------------------------------------------------------
# Tab 3: Model Summary
# ------------------------------------------------------------
with tab_summary:
    st.subheader("Model Summary")
    st.write(f"""
    **Model:** XGBoost Regressor  
    **Training Samples:** {TRAIN_SAMPLES:,}  
    **Features:** {len(feature_order)}  

    ### Performance
    - **R²:** {R2}
    - **MAPE:** {MAPE}%
    - **Median APE:** {MdAPE}%

    The model predicts log-transformed sale prices and
    converts them back to dollar values via `exp`.
    """)