import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

try:
    knn_model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError(
        "model.pkl or scaler.pkl not found. "
        "Please run 'python train_model.py' first to train and save the model."
    )

SPECIES_NAMES = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "No input data received."}), 400

    required_fields = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

    values = []
    for field in required_fields:
        raw_value = data.get(field)
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            return jsonify({"error": f"'{field}' must be a valid number."}), 400

        if numeric_value <= 0:
            return jsonify({"error": f"'{field}' must be a positive number."}), 400

        if numeric_value > 50:
            return jsonify({"error": f"'{field}' value seems unrealistic. Please check your input."}), 400

        values.append(numeric_value)

    input_array = np.array(values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    predicted_class = knn_model.predict(input_scaled)[0]
    predicted_species = SPECIES_NAMES[predicted_class]

    probabilities = knn_model.predict_proba(input_scaled)[0]
    confidence = round(float(probabilities[predicted_class]) * 100, 2)

    return jsonify({
        "prediction": predicted_species,
        "confidence": confidence,
        "probabilities": {
            SPECIES_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(probabilities)
        }
    })


if __name__ == "__main__":
    app.run(debug=True)
