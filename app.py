import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import pywt

# Function to apply DWT (copied from your original code)
def apply_dwt(image, wavelet='haar', threshold_factor=0.04):
    coeffs = pywt.dwt2(image, wavelet)
    cA, (cH, cV, cD) = coeffs

    threshold = threshold_factor * np.max([np.max(np.abs(cH)), np.max(np.abs(cV)), np.max(np.abs(cD))])

    def thresholding(coef, threshold):
        return np.where(np.abs(coef) > threshold, coef, 0)

    cH_thresh = thresholding(cH, threshold)
    cV_thresh = thresholding(cV, threshold)
    cD_thresh = thresholding(cD, threshold)

    denoised_image = pywt.idwt2((cA, (cH_thresh, cV_thresh, cD_thresh)), wavelet)
    denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)

    dwt_features = np.concatenate([cA.flatten(), cH_thresh.flatten(), cV_thresh.flatten(), cD_thresh.flatten()])
    dwt_features = dwt_features / np.max(dwt_features)
    dwt_image = dwt_features.reshape((96, 96))

    return denoised_image, dwt_image

# Function to preprocess the uploaded image
def preprocess_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_resize = cv2.resize(img_gray, (96, 96))
    _, dwt_image = apply_dwt(img_resize)
    img_final = cv2.resize(dwt_image, (32, 64))
    return img_final / 255.0

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model('full_true_upgraded_age_gender_model.h5')

# Main Streamlit app
def main():
    st.title('Fingerprint Analysis App')
    st.write('Upload a fingerprint image to predict gender and age group.')

    # File uploader
    uploaded_file = st.file_uploader("Choose a fingerprint image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read and preprocess the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        preprocessed_image = preprocess_image(image)

        # Display the uploaded image
        st.image(image, caption='Uploaded Fingerprint', use_column_width=True)

        # Load the model and make predictions
        model = load_trained_model()
        gender_pred, age_pred = model.predict(np.expand_dims(preprocessed_image, axis=0))

        # Process predictions
        gender = "Female" if gender_pred[0][1] > 0.5 else "Male"
        age_group_map = {
            0: '0-10 years', 1: '11-20 years', 2: '21-30 years', 3: '31-40 years',
            4: '41-50 years', 5: '51-60 years', 6: '61-70 years', 7: '71-80 years',
            8: '81-90 years', 9: '91-100 years'
        }
        age_group = age_group_map[np.argmax(age_pred[0])]

        # Display results
        st.subheader('Prediction Results:')
        st.write(f"Predicted Gender: {gender}")
        st.write(f"Predicted Age Group: {age_group}")

        # Display confidence scores
        st.subheader('Confidence Scores:')
        st.write(f"Gender Confidence: {max(gender_pred[0]) * 100:.2f}%")
        st.write(f"Age Group Confidence: {max(age_pred[0]) * 100:.2f}%")

if __name__ == '__main__':
    main()
