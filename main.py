from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Disease info
CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
DISEASE_NAMES = {
    'MEL': 'Melanoma',
    'NV': 'Melanocytic Nevi (Mole)',
    'BCC': 'Basal Cell Carcinoma',
    'AKIEC': 'Actinic Keratosis',
    'BKL': 'Benign Keratosis',
    'DF': 'Dermatofibroma',
    'VASC': 'Vascular Lesion'
}
DESCRIPTIONS = {
    'MEL': 'A serious form of skin cancer. Please consult a dermatologist immediately.',
    'NV': 'A common benign mole. Usually harmless but monitor for changes.',
    'BCC': 'Most common skin cancer. Rarely spreads but needs treatment.',
    'AKIEC': 'Pre-cancerous lesion caused by sun damage. See a doctor.',
    'BKL': 'Non-cancerous skin growth. Usually harmless.',
    'DF': 'Benign fibrous skin nodule. Usually harmless.',
    'VASC': 'Vascular skin lesion. Usually benign.'
}

# ✅ Load model
print("Loading AI model...")
model = tf.keras.models.load_model("../ai-model/model/skin_disease_model.h5")
print("✅ Model loaded!")

@app.get("/")
def home():
    return {"message": "Skin Disease Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index]) * 100
    predicted_class = CLASSES[predicted_index]

    return {
        "disease_code": predicted_class,
        "disease_name": DISEASE_NAMES[predicted_class],
        "confidence": round(confidence, 2),
        "description": DESCRIPTIONS[predicted_class],
        "disclaimer": "This is not a substitute for professional medical advice."
    }