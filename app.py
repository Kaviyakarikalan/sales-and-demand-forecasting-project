import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# 1. Load model
# -----------------------------
@st.cache_data  # caches the model for faster reloads
def load_model():
    return joblib.load("sales_forecast_model.pkl")

model = load_model()

# -----------------------------
# 2. App title and description
# -----------------------------
st.title("Amazon Product Price Predictor Dashboard")
st.write(
    "Enter product details below to predict the estimated price of an Amazon product."
)

# -----------------------------
# 3. Input fields
# -----------------------------
discount = st.number_input(
    "Discount Percentage (%)", min_value=0, max_value=100, step=1, value=0
)
rating = st.number_input(
    "Product Rating (0.0 - 5.0)", min_value=0.0, max_value=5.0, step=0.1, value=3.0
)
rating_count = st.number_input(
    "Rating Count", min_value=0, step=1, value=10
)

# -----------------------------
# 4. Make prediction
# -----------------------------
if st.button("Predict Price"):
    # Prepare input
    input_features = [[discount, rating, rating_count]]
    
    # Predict
    predicted_price = model.predict(input_features)[0]
    
    # Show prediction
    st.success(f"Predicted Price: ₹{predicted_price:.2f}")
    
    # -----------------------------
    # Step 3: Feature importance visualization
    # -----------------------------
    feature_importances = pd.DataFrame({
        "Feature": ["Discount (%)", "Rating", "Rating Count"],
        "Importance": model.feature_importances_
    })
    st.subheader("Feature Importance")
    st.bar_chart(feature_importances.set_index("Feature"))
    
    # -----------------------------
    # Step 4: Download button
    # -----------------------------
    results_df = pd.DataFrame({
        "Discount (%)": [discount],
        "Rating": [rating],
        "Rating Count": [rating_count],
        "Predicted Price": [predicted_price]
    })

    st.download_button(
        label="Download Prediction",
        data=results_df.to_csv(index=False),
        file_name="prediction.csv",
        mime="text/csv"
    )

# -----------------------------
# Optional: Footer / Notes
# -----------------------------
st.markdown(
    """
    ---
    This dashboard predicts Amazon product prices based on discount, rating, and rating count.
    """
)