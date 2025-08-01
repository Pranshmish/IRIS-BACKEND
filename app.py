from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "https://iris-azure-ten.vercel.app"])
  # Allow requests only from React dev server

# Load the ML model
try:
    model = joblib.load("model/model.pkl")  # Ensure this path is correct
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask server running ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        # Extract and convert input features
        features = [
            float(data["sepalLength"]),
            float(data["sepalWidth"]),
            float(data["petalLength"]),
            float(data["petalWidth"])
        ]

        prediction = model.predict([features])[0]
        return jsonify({"prediction": str(prediction)})

    except KeyError as ke:
        return jsonify({"error": f"Missing field: {ke}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
