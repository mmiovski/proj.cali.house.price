# libraries and packages
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# custom CSS for styling
st.markdown("""
<style>

div.stButton > button:first-child {
    background-color: #4B9CD3;
    color: white;
    padding: 0.75rem 2rem;
    font-size: 1.1rem;
    font-weight: 600;
    border-radius: 6px;
}

div.stButton > button:first-child:hover {
    background-color: #3a8ac4;
}

.prediction-price {
    font-size: 1.4rem;
    font-weight: 600;
}

.confidence-text {
    font-size: 1.1rem;
    font-weight: 400;
}

</style>
""", unsafe_allow_html=True)

# load model and features
model = joblib.load("final_xgb_nolist.pkl")
with open("features_nolist.json", "r") as f:
    feature_order = json.load(f)

# current model metrics (update dynamically later) 
R2 = 0.89
MAPE = 10.54
MdAPE = 7.07
TRAIN_SAMPLES = 44142

# pretty names for chart
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

# page settings
st.set_page_config(
    page_title="California Home Price Predictor",
    layout="wide"
)

st.title("California Home Price Prediction")

# tabs
tab_pred, tab_importance, tab_summary = st.tabs(
    ["Prediction", "Feature Importance", "Model Summary"]
)

###
# tab 1: prediction
###

with tab_pred:

    st.subheader("Enter Property Details")

    with st.form("prediction_form"):

        # LOCATION
        st.markdown("#### Location")
        c1, c2 = st.columns(2)

        with c1:
            Latitude = st.number_input(
                "Latitude",
                min_value=32.0, max_value=42.0, step=0.001,
                help="California latitude typically ranges from 32–42."
            )

        with c2:
            Longitude = st.number_input(
                "Longitude",
                min_value=-124.0, max_value=-114.0, step=0.001,
                help="California longitude typically ranges from -124 to -114."
            )

        # SIZE
        st.markdown("#### Size")
        c3, c4 = st.columns(2)

        with c3:
            LivingArea = st.number_input(
                "Living Area (sqft)",
                min_value=1.0, step=50.0,
                help="Total finished living area (sqft)."
            )

        with c4:
            LotSizeAcres = st.number_input(
                "Lot Size (acres)",
                min_value=0.0, step=0.01,
                help="Total lot size in acres."
            )

        Stories = st.number_input(
            "Stories",
            min_value=1.0, step=0.5,
            help="Total building stories including partial floors."
        )

        # BED & BATH
        st.markdown("#### Bedrooms & Bathrooms")
        c5, c6 = st.columns(2)

        with c5:
            BedroomsTotal = st.number_input("Bedrooms", min_value=1, step=1)

        with c6:
            BathroomsTotalInteger = st.number_input("Bathrooms", min_value=0.5, step=0.5)

        # FEATURES
        st.markdown("#### Home Features")
        c7, c8 = st.columns(2)

        with c7:
            FireplaceYN = st.selectbox("Fireplace", ["No", "Yes"])

        with c8:
            PoolPrivateYN = st.selectbox("Private Pool", ["No", "Yes"])

        ParkingTotal = st.number_input("Total Parking", min_value=0, step=1)

        # MARKET
        st.markdown("#### Market Info")
        c9, c10 = st.columns(2)

        with c9:
            DaysOnMarket = st.number_input("Days on Market", min_value=0, step=1)

        with c10:
            YearBuilt = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=pd.Timestamp.now().year
            )

        submitted = st.form_submit_button("Predict Price")

    # VALIDATION
    def validate():
        errors = []
        if BathroomsTotalInteger > BedroomsTotal + 3:
            errors.append("Bathrooms unusually high relative to bedrooms.")

        if LotSizeAcres > 0 and LivingArea > LotSizeAcres * 43560:
            errors.append("Living area exceeds lot size (sqft).")

        return errors

    # ON SUBMISSION
    if submitted:
        errors = validate()
        if errors:
            st.error("Please fix the following issues:")
            for e in errors:
                st.write(f"- {e}")
            st.stop()

        bool_map = {"No": 0, "Yes": 1}
        FireplaceYN = bool_map[FireplaceYN]
        PoolPrivateYN = bool_map[PoolPrivateYN]

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

        # Predict log-price and convert back
        y_log = model.predict(row)[0]
        y_pred = np.expm1(y_log)

        # Confidence Interval via MdAPE
        lower = y_pred / (1 + MdAPE / 100)
        upper = y_pred * (1 + MdAPE / 100)

        st.markdown(
            f"<p class='prediction-price'>Estimated Close Price: <strong>${y_pred:,.0f}</strong></p>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<p class='confidence-text'>Confidence Interval: ${lower:,.0f} — ${upper:,.0f}</p>",
            unsafe_allow_html=True
        )

###
# tab 2: feature importance
###

with tab_importance:

    st.subheader("Feature Importance (XGBoost)")

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    clean_features = [
        pretty_names.get(f, f)
        for f in np.array(feature_order)[sorted_idx]
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(clean_features, importances[sorted_idx], color="#4B9CD3")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)

### 
# tab 3: model summary
### 

with tab_summary:

    st.subheader("Model Summary")

    st.write(f"""
    **Model Type:** XGBoost Regressor  
    **Region:** California MLS  
    **Features Used:** 12  
    **Training Samples:** {TRAIN_SAMPLES:,}  

    ### Performance  
    - **R²:** {R2}  
    - **MAPE:** {MAPE}%  
    - **Median APE:** {MdAPE}%  

    This model predicts **log-transformed sale price** and applies an exponential transformation
    to return dollar-valued estimates.
    """)