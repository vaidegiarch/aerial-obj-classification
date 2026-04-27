import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------------
# Load model (safe + cached)
# -------------------------------
@st.cache_resource
def load_my_model():
    try:
        model = load_model("best_cnn_model.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_my_model()

# -------------------------------
# UI
# -------------------------------
st.title("🦅 Bird vs 🚁 Drone Classifier")
st.write("Upload an image to classify whether it's a Bird or a Drone")

# -------------------------------
# Prediction with TTA (improved)
# -------------------------------
def predict_tta(model, img_array):
    preds = []

    # Original
    preds.append(model.predict(img_array, verbose=0)[0][0])

    # Horizontal flip
    flipped = np.flip(img_array, axis=2)
    preds.append(model.predict(flipped, verbose=0)[0][0])

    # Vertical flip (NEW - improves stability)
    v_flipped = np.flip(img_array, axis=1)
    preds.append(model.predict(v_flipped, verbose=0)[0][0])

    # Rotate 90°
    rotated1 = np.rot90(img_array, k=1, axes=(1, 2))
    preds.append(model.predict(rotated1, verbose=0)[0][0])

    # Rotate 180°
    rotated2 = np.rot90(img_array, k=2, axes=(1, 2))
    preds.append(model.predict(rotated2, verbose=0)[0][0])

    return np.mean(preds)

# -------------------------------
# Upload
# -------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------------
    # Preprocess (robust)
    # -------------------------------
    img = image.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # -------------------------------
    # Predict
    # -------------------------------
    prediction = predict_tta(model, img)

    st.write(f"Raw prediction score: {prediction:.3f}")

    # -------------------------------
    # Smarter classification logic
    # -------------------------------
    if prediction > 0.65:
        label = "🚁 Drone"
        confidence = prediction
    elif prediction < 0.35:
        label = "🦅 Bird"
        confidence = 1 - prediction
    else:
        label = "⚠️ Not Sure"
        confidence = max(prediction, 1 - prediction)

    # -------------------------------
    # Output
    # -------------------------------
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}")
