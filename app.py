import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)

# ----------------------------
# Load model and scaler
# ----------------------------
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------------
# Features (all numeric)
# ----------------------------
FEATURES = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt", "Location", "Condition", "Garage"]
NUMERIC_FEATURES = FEATURES  # all are numeric now

# ----------------------------
# Prediction function
# ----------------------------
def predict_price(form_data):
    # Parse values (expect numeric strings)
    data = {}
    for feature in FEATURES:
        if feature not in form_data:
            raise ValueError(f"Missing feature: {feature}")
        # allow int/float in JSON too
        data[feature] = float(form_data[feature])

    df = pd.DataFrame([data], columns=FEATURES)

    # Scale numeric values
    df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])

    # Predict using numpy array to avoid feature-name checks
    prediction = model.predict(df.values)
    return round(float(prediction[0]), 2)

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    price = None
    error = None
    last_input = {}
    if request.method == "POST":
        try:
            last_input = request.form.to_dict()
            price = predict_price(last_input)
        except Exception as e:
            error = str(e)

    # Pass suggestion text for the numeric codes to the template
    meta = {
        "location_codes": "1=Downtown, 2=Suburban, 3=Urban, 4=Rural",
        "condition_codes": "1=Excellent, 2=Good, 3=Fair, 4=Poor",
        "garage_codes": "1=Yes, 0=No"
    }

    return render_template(
        "index.html",
        prediction=price,
        error=error,
        last_input=last_input,
        meta=meta
    )

@app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()
        price = predict_price(data)
        return jsonify({"price": price})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)


