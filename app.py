import os
import numpy as np
import PIL.ImageOps
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="MNIST Digit Predictor", layout="centered")

st.title("MNIST Digit Predictor")
st.write("Upload an image of a handwritten digit to get a prediction.")

MODEL_PATH = "67102010529_mnist_model.keras"

@st.cache_resource
def load_mnist_model(path: str):
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)

model = load_mnist_model(MODEL_PATH)

if model is None:
    st.error(f"Model file '{MODEL_PATH}' not found. Please ensure the file is in the working directory.")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("L")

        invert_colors = st.checkbox("Invert colors (MNIST requires white digit on black background)", value=True)
        if invert_colors:
            img = PIL.ImageOps.invert(img)

        img_resized = img.resize((28, 28))
        img_array = np.array(img_resized).astype("float32") / 255.0
        input_tensor = np.expand_dims(img_array, axis=(0, -1))

        prediction = model.predict(input_tensor)[0]
        predicted_digit = int(np.argmax(prediction))
        confidence = float(prediction[predicted_digit]) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_resized, caption="Processed Model Input (28x28)", use_container_width=True)
        with col2:
            st.success(f"Predicted Digit: **{predicted_digit}**")
            st.metric(label="Confidence", value=f"{confidence:.2f}%")

        st.subheader("Prediction Probabilities")
        st.bar_chart(prediction)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
