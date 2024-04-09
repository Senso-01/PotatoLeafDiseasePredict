import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("model_2.h5")

# Define class names
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']  # Update with your class names

# Function to preprocess the image
def preprocess_image(img):
    img = img / 255.0  # Normalize pixel values
    return img

# Function to make predictions
def predict(image):
    # Preprocess the image
    img = preprocess_image(image)
    img_array = tf.expand_dims(img, 0)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

def main():
    st.title("TensorFlow Model Deployment with Streamlit")
    st.sidebar.title("Options")
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # Upload image through Streamlit
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image_to_show = image.load_img(uploaded_file, target_size=(256, 256))
        st.image(image_to_show, caption="Uploaded Image.", use_column_width=True)

        # Make prediction on the uploaded image
        if st.button("Predict"):
            predicted_class, confidence = predict(image.img_to_array(image_to_show))
            st.success(f"Predicted Class: {predicted_class}, Confidence: {confidence}%")

if __name__ == "__main__":
    main()
