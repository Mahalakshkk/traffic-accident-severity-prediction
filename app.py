from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load saved model & preprocessors
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""
    severity_class = ""

    if request.method == "POST":
        # Get form values
        weather = request.form["weather"]
        visibility = float(request.form["visibility"])
        temperature = float(request.form["temperature"])
        wind = float(request.form["wind"])
        precipitation = float(request.form["precipitation"])
        sunrise = int(request.form["sunrise"])

        # Encode categorical feature
        weather_encoded = encoder.transform([weather])[0]

        # Arrange input in SAME order as training
        input_data = np.array([[
            weather_encoded,
            visibility,
            temperature,
            wind,
            precipitation,
            sunrise
        ]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        result = model.predict(input_scaled)[0]

        # Severity mapping (label + CSS class)
        severity_map = {
            0: ("Low", "low"),
            1: ("Moderate", "moderate"),
            2: ("High", "high"),
            3: ("Critical", "critical")
        }

        label, severity_class = severity_map[result]
        prediction_text = f"Predicted Accident Severity: {label}"

    return render_template(
        "index.html",
        prediction=prediction_text,
        severity_class=severity_class
    )

if __name__ == "__main__":
    app.run(debug=True)
