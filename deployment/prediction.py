import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

def run():
    # Load the saved model
    model = tf.keras.models.load_model("best_model.h5")

    # Define the label names of class
    label_names = ["Beer Bottles", "Plastic Bottles", "Soda Bottles",'Water Bottle','Wine Bottle']
    # Define the Streamlit app
    st.title("BOttle Detection")
    st.write("Choose an image want to classify.")

    # Allow the user to select an image file
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the image using TensorFlow
        img = tf.keras.utils.load_img(uploaded_file, target_size=(150, 150, 3))

        # Convert the PIL.Image.Image object to a NumPy array
        x = tf.keras.utils.img_to_array(img)

        # Expand the array to add a batch dimension
        x = np.expand_dims(x, axis=0)

        # Normalize the image data
        x = x / 255.0

        # Make the prediction using the loaded model
        y_pred = model.predict(x)

        # Get the index of the predicted class with the highest probability
        class_idx = np.argmax(y_pred, axis=1)[0]

        # Display the predicted class label and image to the user
        st.write(f"Detection for uploaded image: {label_names[class_idx]}")
        st.image(img, caption=f"{label_names[class_idx]}", use_column_width=True)
        

if __name__ == "__app__":
    run()