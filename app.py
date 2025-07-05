from flask import Flask, request, jsonify
import joblib
import gdown
import numpy as np
import pandas as pd
from PIL import Image
import os
import io

app = Flask(__name__)

# Emplacements des fichiers
xgb_path = "model/model_xgboost.pkl"
resnet_path = "model/resnet50_model.keras"
encoder_path = "tabular_encoder.pkl"
features_path = "tabular_features.pkl"

# Variables globales
xgb_model = None
resnet_model = None
encoder = None
feature_names = None


def download_and_load_models():
    global xgb_model, resnet_model, encoder, feature_names

    os.makedirs("model", exist_ok=True)

    # Télécharger les modèles s'ils n'existent pas
    if not os.path.exists(xgb_path):
        gdown.download("https://drive.google.com/uc?id=1v6kqLaFlTJNFfLEk6CBCDhQmamm6BCZx", xgb_path, quiet=False)

    if not os.path.exists(resnet_path):
        gdown.download("https://drive.google.com/uc?id=1YcgcCaDBvydz1gTXRra6cF8s9reQmomv", resnet_path, quiet=False)

    # Charger le modèle XGBoost
    xgb_model = joblib.load(xgb_path)

    # Charger le modèle ResNet dynamiquement (important pour Heroku)
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    globals()['resnet_model'] = load_model(resnet_path)

    # Charger les encodeurs
    encoder = joblib.load(encoder_path)
    feature_names = joblib.load(features_path)

    print("✅ Tous les modèles sont chargés.")


@app.route('/predict/tabulaire', methods=['POST'])
def predict_tabulaire():
    global xgb_model, encoder, feature_names

    data = request.get_json()
    df = pd.DataFrame([data])

    X_encoded = encoder.transform(df)
    X_df = pd.DataFrame(X_encoded, columns=feature_names)

    prediction = xgb_model.predict(X_df)[0]
    return jsonify({"prediction": "Fraude" if prediction == 1 else "Non-Fraude"})


@app.route('/predict/image', methods=['POST'])
def predict_image():
    global resnet_model

    if 'image' not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    img_file = request.files['image']
    img = Image.open(io.BytesIO(img_file.read())).convert("RGB")
    img = img.resize((224, 224))

    # Importer TensorFlow ici uniquement pour éviter le problème de build
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image

    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = resnet_model.predict(img_array)[0][0]
    result = "Fraude" if prediction >= 0.3 else "Non-Fraude"
    return jsonify({"prediction": result, "probabilité_fraude": float(prediction)})


if __name__ == '__main__':
    download_and_load_models()
    port = int(os.environ.get("PORT", 5000))  
    app.run(host='0.0.0.0', port=port)
