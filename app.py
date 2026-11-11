import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("🩺 Skin Cancer Detection App (9-Class)")
st.write("Upload a skin lesion image to predict (Demo Purpose Only)")

# Load model
model = load_model(r"C:\Users\arsha\OneDrive\Desktop\DL1\newDL1.keras")

# 9-class labels (HAM10000)
labels = [
    'Actinic keratosis',
    'Basal cell carcinoma',
    'Benign keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Nevus',
    'Pigmented benign keratosis',
    'Seborrheic keratosis',
    'Vascular lesion'
]

# Show cancers model can detect
st.subheader("🧬 Cancers this model can detect:")
for i, label in enumerate(labels, 1):
    st.write(f"**{i}. {label}**")

# File uploader
uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Predict
    preds = model.predict(x)

    # Debug info
    argmax_index = int(np.argmax(preds))
    st.write("### 🔍 Debug Info")
    st.write("Predictions Shape:", preds.shape)
    st.write("Argmax Index:", argmax_index)
    st.write("Labels Count:", len(labels))

    # Final prediction
    pred_class = labels[argmax_index]
    confidence = float(np.max(preds)) * 100

    st.success(f"### ✅ Prediction: **{pred_class}** ({confidence:.2f}%)")