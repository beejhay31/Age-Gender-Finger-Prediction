import streamlit as st
import cv2
import numpy as np
import pywt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model_path = 'age_gender_model(7).h5'
model = load_model(model_path)

# Define the specific age ranges for your clusters
age_ranges = {
    0: '0-10',
    1: '11-20',
    2: '21-30',
    3: '31-40',
    4: '41-50',
    5: '51-60',
    6: '61-70',
    7: '71-80',
    8: '81-90',
    9: '91-100'
}

# Updated DWT function with denoising, edge detection, and feature extraction
def apply_dwt(image, wavelet='haar', threshold_factor=0.04):
    # Apply DWT
    coeffs = pywt.dwt2(image, wavelet)
    cA, (cH, cV, cD) = coeffs

    # Denoising: Apply thresholding to detail coefficients
    threshold = threshold_factor * np.max([np.max(np.abs(cH)), np.max(np.abs(cV)), np.max(np.abs(cD))])

    def thresholding(coef, threshold):
        return np.where(np.abs(coef) > threshold, coef, 0)

    cH_thresh = thresholding(cH, threshold)
    cV_thresh = thresholding(cV, threshold)
    cD_thresh = thresholding(cD, threshold)

    # Reconstruct denoised image
    denoised_image = pywt.idwt2((cA, (cH_thresh, cV_thresh, cD_thresh)), wavelet)
    denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)

    # Edge Detection
    edges_horizontal = np.abs(cH_thresh)
    edges_vertical = np.abs(cV_thresh)
    edges_diagonal = np.abs(cD_thresh)

    # Feature Extraction: Flatten and concatenate coefficients
    dwt_features = np.concatenate([cA.flatten(), cH_thresh.flatten(), cV_thresh.flatten(), cD_thresh.flatten()])
    dwt_features = dwt_features / np.max(dwt_features)  # Normalize features

    # Reshape to match model input (64x32)
    dwt_image = dwt_features.reshape((64, 32))

    return denoised_image, edges_horizontal, edges_vertical, edges_diagonal, dwt_image

# Streamlit app interface
st.title("Age and Gender Prediction System Using Fingerprint Images")
st.write("This app predicts **age range** and **gender** from fingerprint images.")

# Add sidebar for user details
st.sidebar.header("User Information")
name = st.sidebar.text_input("Name")
id_number = st.sidebar.text_input("ID Number")
st.sidebar.write(f"User: {name} (ID: {id_number})")

# Add columns for layout similar to the image you uploaded
col1, col2 = st.columns(2)

# Upload image in the first column
with col1:
    uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["bmp", "jpg", "jpeg", "png"])

# Placeholder for image and predictions in the second column
with col2:
    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image
        image_resized = cv2.resize(gray_image, (64, 32))

        # Apply the updated DWT function
        denoised_image, edges_horizontal, edges_vertical, edges_diagonal, dwt_image = apply_dwt(image_resized)

        # Expand dimensions of the DWT image for model input
        dwt_image = np.expand_dims(dwt_image, axis=-1)  # Add channel dimension
        dwt_image = np.expand_dims(dwt_image, axis=0)   # Add batch dimension

        # Predict age and gender
        age_pred, gender_pred = model.predict(dwt_image)

        # Decode predictions
        age_range_class = np.argmax(age_pred)
        age_range = age_ranges[age_range_class]
        gender = 'Male' if gender_pred < 0.5 else 'Female'

        # Display the original image
        st.image(image, caption="Uploaded Fingerprint Image", use_column_width=True)

        # Display the predictions
        st.write(f"**Predicted Age Range:** {age_range}")
        st.write(f"**Predicted Gender:** {gender}")

        # Add an expander to show the processed images
        with st.expander("View Processed Images"):
            # Display the denoised image as it is
            st.write("**Denoised Image**")
            st.image(denoised_image, caption="Denoised Fingerprint Image", use_column_width=True)

            # Normalize the edge arrays before displaying them
            edges_horizontal_normalized = edges_horizontal / np.max(edges_horizontal)
            edges_vertical_normalized = edges_vertical / np.max(edges_vertical)
            edges_diagonal_normalized = edges_diagonal / np.max(edges_diagonal)

            st.write("**Horizontal Edges**")
            st.image(edges_horizontal_normalized, caption="Horizontal Edges", use_column_width=True)

            st.write("**Vertical Edges**")
            st.image(edges_vertical_normalized, caption="Vertical Edges", use_column_width=True)

            st.write("**Diagonal Edges**")
            st.image(edges_diagonal_normalized, caption="Diagonal Edges", use_column_width=True)
