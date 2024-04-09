import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

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
    st.title("Potato Plant Disease Prediction using TensorFlow by Senso-01")
    st.sidebar.title("Upload your Image")
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # Add file uploader for local file choosing
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Add camera input
    use_camera = st.sidebar.checkbox("Use Camera")

    if use_camera:
        # Capture video from the camera
        video_capture = cv2.VideoCapture(0)

        # Continuously read frames from the camera
        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.error("Error: Unable to capture video.")
                break

            # Display the frame in the Streamlit app
            st.image(frame, channels="BGR", use_column_width=True)
            if st.button("Predict"):
                predicted_class, confidence = predict(frame)
                st.success(f"Predicted Class: {predicted_class}, Confidence: {confidence}%")
            if st.button("Stop"):
                break

    elif uploaded_file is not None:
        # Display the uploaded image
        image_to_show = image.load_img(uploaded_file, target_size=(256, 256))
        st.image(image_to_show, caption="Uploaded Image.", use_column_width=True)

        # Make prediction on the uploaded image
        if st.button("Predict"):
            predicted_class, confidence = predict(image.img_to_array(image_to_show))
            st.success(f"Predicted Class: {predicted_class}, Confidence: {confidence}%")

if __name__ == "__main__":
    main()
