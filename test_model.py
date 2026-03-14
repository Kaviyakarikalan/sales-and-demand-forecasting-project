import pandas as pd
import joblib

# Load saved model
model = joblib.load("sales_forecast_model.pkl")

# Example new product data
sample = pd.DataFrame({
    "discount_percentage": [50],
    "rating": [4.3],
    "rating_count": [1200]
})

prediction = model.predict(sample)
print("Predicted price for sample product:", prediction)