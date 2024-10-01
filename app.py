import streamlit as st
# Set page config at the very beginning
st.set_page_config(page_title="Fingerprint Analysis App", layout="wide")

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import pywt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the saved model
@st.cache_resource
def load_model_and_kmeans():
    model = load_model('final_fingerprint_model_kfold.h5', compile=False)
    kmeans = KMeans(n_clusters=10, random_state=42)
    return model, kmeans

model, kmeans = load_model_and_kmeans()

# Define the specific age ranges for your clusters
age_ranges = {
    0: '0-10', 1: '11-20', 2: '21-30', 3: '31-40', 4: '41-50',
    5: '51-60', 6: '61-70', 7: '71-80', 8: '81-90', 9: '91-100'
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
    dwt_image = dwt_features.reshape((96, 96))

    return denoised_image, edges_horizontal, edges_vertical, edges_diagonal, dwt_image

def process_image(image):
    img_array = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img_array, (96, 96))
    return apply_dwt(img_resized)

def display_processed_images(images, titles):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    for ax, img, title in zip(axs, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    st.pyplot(fig)

def main():
    #st.set_page_config(page_title="Fingerprint Analysis App", layout="wide")

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #e0e0e0;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("🖐️ Fingerprint Analysis App")

    # User information input
    col1, col2 = st.columns(2)
    with col1:
        user_name = st.text_input("Enter your name:")
    with col2:
        matric_number = st.text_input("Enter your matric number:")

    if user_name and matric_number:
        st.success(f"Welcome, {user_name}! (Matric Number: {matric_number})")

    uploaded_file = st.file_uploader("Choose a fingerprint image...", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("")
        
        if st.button("Analyze Fingerprint"):
            st.write("Analyzing...")

            # Process the image
            denoised_image, edges_horizontal, edges_vertical, edges_diagonal, dwt_image = process_image(uploaded_file)

            # Display all processed images
            display_processed_images(
                [denoised_image, edges_horizontal, edges_vertical, edges_diagonal, dwt_image],
                ["Denoised Image", "Horizontal Edges", "Vertical Edges", "Diagonal Edges", "DWT Features"]
            )

            # Make prediction
            prediction = model.predict(dwt_image.reshape(1, 96, 96, 1))
            gender_pred = prediction[0]
            age_cluster_pred = prediction[1]

            # Get the predicted gender
            gender = "Male" if np.argmax(gender_pred) == 0 else "Female"
            gender_confidence = np.max(gender_pred) * 100

            # Get the predicted age range
            age_cluster = np.argmax(age_cluster_pred)
            age_range = age_ranges[age_cluster]
            age_confidence = np.max(age_cluster_pred) * 100

            # Display results
            st.subheader("Analysis Results:")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Predicted Gender: {gender}")
                st.info(f"Gender Prediction Confidence: {gender_confidence:.2f}%")
            with col2:
                st.info(f"Predicted Age Range: {age_range}")
                st.info(f"Age Prediction Confidence: {age_confidence:.2f}%")


if __name__ == "__main__":
    main()
