from flask import Flask, request, jsonify
import joblib
import gdown
import pandas as pd
import os

app = Flask(__name__)

# Emplacements des fichiers
xgb_path = "model/model_xgboost.pkl"
encoder_path = "tabular_encoder.pkl"
features_path = "tabular_features.pkl"

# Variables globales
xgb_model = None
encoder = None
feature_names = None

@app.before_first_request
def download_and_load_models():
    global xgb_model, encoder, feature_names

    os.makedirs("model", exist_ok=True)

    if not os.path.exists(xgb_path):
        gdown.download("https://drive.google.com/uc?id=1v6kqLaFlTJNFfLEk6CBCDhQmamm6BCZx", xgb_path, quiet=False)

    xgb_model = joblib.load(xgb_path)
    encoder = joblib.load(encoder_path)
    feature_names = joblib.load(features_path)

@app.route('/predict/tabulaire', methods=['POST'])
def predict_tabulaire():
    global xgb_model, encoder, feature_names

    data = request.get_json()
    df = pd.DataFrame([data])

    X_encoded = encoder.transform(df)
    X_df = pd.DataFrame(X_encoded, columns=feature_names)

    prediction = xgb_model.predict(X_df)[0]
    return jsonify({"prediction": "Fraude" if prediction == 1 else "Non-Fraude"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
