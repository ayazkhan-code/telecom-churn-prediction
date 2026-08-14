import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📉", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    top_features = joblib.load("top_features.pkl")
    medians = joblib.load("feature_medians.pkl")
    return model, top_features, medians

model, top_features, medians = load_artifacts()

st.title("📉 Telecom Customer Churn Predictor")
st.write(
    "Predicts the probability that a high-value telecom customer will churn, "
    "using a Random Forest model trained on the top 15 predictive features "
    "(selected via feature importance from a 170+ feature dataset)."
)

st.subheader("Enter customer details")
st.caption("Fields are pre-filled with dataset medians — adjust the ones you know.")

user_input = {}
cols = st.columns(2)
for i, feat in enumerate(top_features):
    default_val = float(medians[feat])
    with cols[i % 2]:
        user_input[feat] = st.number_input(
            feat, value=default_val, format="%.2f"
        )

if st.button("Predict Churn Risk", type="primary"):
    input_df = pd.DataFrame([user_input])[top_features]
    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.divider()
    if pred == 1:
        st.error(f"⚠️ High churn risk — {proba*100:.1f}% probability")
    else:
        st.success(f"✅ Low churn risk — {proba*100:.1f}% probability")
    st.progress(min(int(proba * 100), 100))

st.divider()
st.caption(
    "Full analysis, EDA, and 4-model comparison available in the project notebook. "
    "This demo uses a simplified 15-feature model for usability."
)
