import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model (cached to avoid reloading every time)
@st.cache_resource
def load_my_model():
    return load_model("best_cnn_model.keras")

model = load_my_model()

# Title
st.title("🦅 Bird vs 🚁 Drone Classifier")
st.write("Upload an image to classify whether it's a Bird or a Drone")

# ---- Prediction Function with TTA ----
def predict_tta(model, img_array):
    preds = []

    # Original
    preds.append(model.predict(img_array, verbose=0)[0][0])

    # Horizontal flip
    flipped = np.flip(img_array, axis=2)
    preds.append(model.predict(flipped, verbose=0)[0][0])

    # Rotate 90°
    rotated1 = np.rot90(img_array, k=1, axes=(1, 2))
    preds.append(model.predict(rotated1, verbose=0)[0][0])

    # Rotate 180°
    rotated2 = np.rot90(img_array, k=2, axes=(1, 2))
    preds.append(model.predict(rotated2, verbose=0)[0][0])

    return np.mean(preds)

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---- Preprocess ----
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # ---- Predict ----
    prediction = predict_tta(model, img)

    st.write(f"Raw prediction score: {prediction:.3f}")

    # ---- Classification Logic ----
    if prediction > 0.75:
        label = "🚁 Drone"
        confidence = prediction
    elif prediction < 0.30:
        label = "🦅 Bird"
        confidence = 1 - prediction
    else:
        label = "⚠️ Not Sure"
        confidence = prediction

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}")
