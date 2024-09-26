import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('C:/my_model_food101.keras')

# Class labels (replace these with the actual labels for your Food-101 dataset)
class_labels = ['apple_pie', 'baby_back_ribs', 'baklava','beef_carpaccio','beef_tartare','beet_salad','beignets']  # Add all 101 class labels
# Example calorie estimates for some classes (replace with actual values)
calorie_dict = {
    'apple_pie': 237,  # calories per serving
    'baby_back_ribs': 350,
    'baklava': 540,
    'beef_carpaccio':460,
    'beef_tartare':560,
    'beet_salad':450,
    'beignets':340

}

# Streamlit app
st.title("Food-101 Image Classification")
st.write("Upload an image of a food item and the model will predict its class!")

# File uploader for image input
uploaded_file = st.file_uploader("Choose a food image...", type="jpg")
if uploaded_file is not None:
    # Load the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = img.resize((150, 150))  # Resize the image to 150x150 (same size as training data)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Rescale pixel values

    # Predict the class
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    confidence = np.max(predictions) * 100

    # Estimate calories
    estimated_calories = calorie_dict.get(predicted_label, "N/A")  # Get the calorie estimate or "N/A" if not available

    # Display the prediction and estimated calories
    st.write(f"Prediction: **{predicted_label}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
    st.write(f"Estimated Calories: **{estimated_calories} kcal**")