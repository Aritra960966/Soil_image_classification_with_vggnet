import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the trained model
model = tf.keras.models.load_model("my_model.h5")

# Define class labels (adjust according to your dataset)
class_labels = ["Black ", "Cinder", "Laterite", "Peat", "Yellow"]

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((220, 220))  
    image = np.array(image) / 255.0   
    image = np.expand_dims(image, axis=0)  
    return image

# Streamlit UI
st.title("Soil Type Classification")
st.write("Upload an image to classify the soil type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    # Display result
    st.write(f"### Predicted Soil Type: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
