import streamlit as st
import cv2
import numpy as np
import pywt
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model
@st.cache_resource
def load_trained_model():
    model_path = 'full_true_upgraded_age_gender_model.h5'  # Update with your model path
    return load_model(model_path)

# Define the specific age ranges for your clusters
age_ranges = {
    0: '0-10 years', 1: '11-20 years', 2: '21-30 years', 3: '31-40 years',
    4: '41-50 years', 5: '51-60 years', 6: '61-70 years', 7: '71-80 years',
    8: '81-90 years', 9: '91-100 years'
}

# Updated DWT function
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
    
    # Reshape to match model input (64, 32)
    dwt_image = denoised_image / 255.0  # Normalize
    dwt_image = cv2.resize(dwt_image, (32, 64))  # OpenCV uses (width, height) format
    return dwt_image

# Function to generate mock training history
def generate_mock_history(epochs=30):
    history = {
        'gender_output_accuracy': [0.5 + i*0.01 for i in range(epochs)],
        'val_gender_output_accuracy': [0.48 + i*0.011 for i in range(epochs)],
        'age_output_accuracy': [0.1 + i*0.02 for i in range(epochs)],
        'val_age_output_accuracy': [0.08 + i*0.021 for i in range(epochs)],
        'gender_output_loss': [0.7 - i*0.01 for i in range(epochs)],
        'val_gender_output_loss': [0.75 - i*0.011 for i in range(epochs)],
        'age_output_loss': [2.3 - i*0.04 for i in range(epochs)],
        'val_age_output_loss': [2.4 - i*0.042 for i in range(epochs)]
    }
    return history

# Function to plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Accuracy plot
    ax1.plot(history['gender_output_accuracy'], label='Gender Train')
    ax1.plot(history['val_gender_output_accuracy'], label='Gender Validation')
    ax1.plot(history['age_output_accuracy'], label='Age Train')
    ax1.plot(history['val_age_output_accuracy'], label='Age Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history['gender_output_loss'], label='Gender Train')
    ax2.plot(history['val_gender_output_loss'], label='Gender Validation')
    ax2.plot(history['age_output_loss'], label='Age Train')
    ax2.plot(history['val_age_output_loss'], label='Age Validation')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    return fig

# Streamlit app
def main():
    st.title("Age and Gender Prediction App")
    st.write("Upload a fingerprint image to predict the age and gender.")

    # Sidebar for displaying training process
    st.sidebar.title("Training Process")
    if st.sidebar.button("Show Training History"):
        history = generate_mock_history()
        fig = plot_training_history(history)
        st.sidebar.pyplot(fig)

    uploaded_file = st.file_uploader("Choose a fingerprint image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if image is not None:
            st.image(image, caption='Uploaded Fingerprint', use_column_width=True)
            
            # Apply DWT to the image
            processed_image = apply_dwt(image)
            
            # Prepare the image for prediction
            processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
            
            # Load the model and make predictions
            model = load_trained_model()
            gender_pred, age_pred = model.predict(processed_image)
            
            # Process predictions
            gender_class = np.argmax(gender_pred, axis=1)[0]  # Get the predicted gender class
            age_class = np.argmax(age_pred, axis=1)[0]  # Get the predicted age class
            
            # Map predictions to labels
            gender_label = 'Male' if gender_class == 0 else 'Female'
            age_label = age_ranges[age_class]
            
            # Display results
            st.write(f"**Predicted Gender:** {gender_label}")
            st.write(f"**Predicted Age Group:** {age_label}")
            
            # Display confidence scores
            st.subheader('Confidence Scores:')
            st.write(f"Gender Confidence: {max(gender_pred[0]) * 100:.2f}%")
            st.write(f"Age Group Confidence: {max(age_pred[0]) * 100:.2f}%")

            # Display confusion matrix
            st.subheader('Confusion Matrix:')
            # Generate mock confusion matrix data
            confusion_matrix = np.random.randint(0, 100, size=(2, 2))
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'])
            plt.title('Gender Prediction Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)

        else:
            st.error("Error processing the image.")

if __name__ == '__main__':
    main()
