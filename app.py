import streamlit as st
import cv2
import numpy as np
import pywt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model_path = 'age-genderKnn.h5'
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

    edges_horizontal = np.abs(cH_thresh)
    edges_vertical = np.abs(cV_thresh)
    edges_diagonal = np.abs(cD_thresh)

    dwt_features = np.concatenate([cA.flatten(), cH_thresh.flatten(), cV_thresh.flatten(), cD_thresh.flatten()])
    dwt_features = dwt_features / np.max(dwt_features)

# Check the length of dwt_features
    print("Shape of dwt_features:", dwt_features.shape)

    # Adjust the reshape based on the actual size of dwt_features
    # Assuming the total size is 9984, calculate appropriate shape
    dwt_image_size = 64 * 32  # Expected shape
    if dwt_features.size != dwt_image_size:
        raise ValueError(f"Expected {dwt_image_size} elements but got {dwt_features.size}.")

    dwt_image = dwt_features.reshape((64, 32))

    return denoised_image, edges_horizontal, edges_vertical, edges_diagonal, dwt_image
    
    #dwt_image = dwt_features.reshape((64, 32))

    #return denoised_image, edges_horizontal, edges_vertical, edges_diagonal, dwt_image

# Simulate capturing a fingerprint from a scanner
def capture_fingerprint_from_scanner():
    # Replace this code with the actual scanner integration
    # Here, we'll simulate by loading a local image
    scanner_image_path = "scanner_fingerprint.bmp"  # This path would be dynamically created from the scanner
    if os.path.exists(scanner_image_path):
        image = cv2.imread(scanner_image_path, cv2.IMREAD_GRAYSCALE)
        return image
    else:
        st.error("Error: Scanner image not found!")
        return None

# Streamlit interface
st.set_page_config(page_title="Fingerprint Age and Gender Prediction", layout="wide", page_icon=":fingerprint:")

# Custom CSS styling
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        color: #333;
    }
    .stButton>button {
        background-color: #ff6600;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🔍 Fingerprint Age & Gender Prediction System")
st.subheader("Predict the **age range** and **gender** from fingerprint images or a live scanner.")

# Sidebar for user information
with st.sidebar:
    st.header("🔐 User Details")
    name = st.text_input("Name", help="Enter your full name.")
    id_number = st.text_input("ID Number", help="Enter your ID number.")
    st.write(f"👤 **User:** {name}")
    st.write(f"🆔 **ID Number:** {id_number}")

# Columns for layout
col1, col2 = st.columns(2)

# Initialize variables for handling unbound cases
image = None
denoised_image = None
edges_horizontal = None
edges_vertical = None
edges_diagonal = None

# Add options for uploading image or using scanner
option = st.radio("Choose input method:", ("Upload Fingerprint Image", "Use Fingerprint Scanner"))

with col1:
    if option == "Upload Fingerprint Image":
        uploaded_file = st.file_uploader("📁 Upload Fingerprint Image", type=["bmp", "jpg", "jpeg", "png"])
    else:
        st.write("🔌 Ensure your fingerprint scanner is connected.")
        if st.button("Capture Fingerprint from Scanner"):
            image = capture_fingerprint_from_scanner()
            if image is not None:
                st.image(image, caption="Captured Fingerprint Image", use_column_width=True)
            else:
                st.error("Unable to capture fingerprint.")

if option == "Upload Fingerprint Image" and uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image
    image_resized = cv2.resize(gray_image, (64, 32))
    
    # Keep the original image size
    denoised_image, edges_horizontal, edges_vertical, edges_diagonal, dwt_image = apply_dwt(image_resized)

    dwt_image = np.expand_dims(dwt_image, axis=-1)
    dwt_image = np.expand_dims(dwt_image, axis=0)

    age_pred, gender_pred = model.predict(dwt_image)
    age_range_class = np.argmax(age_pred)
    age_range = age_ranges[age_range_class]
    gender = 'Male' if gender_pred < 0.5 else 'Female'

    st.image(image, caption="Uploaded Fingerprint Image", use_column_width=True)
    st.success(f"**Predicted Age Range:** {age_range}")
    st.success(f"**Predicted Gender:** {gender}")

elif option == "Use Fingerprint Scanner" and image is not None:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised_image, edges_horizontal, edges_vertical, edges_diagonal, dwt_image = apply_dwt(gray_image)
    dwt_image = np.expand_dims(dwt_image, axis=-1)
    dwt_image = np.expand_dims(dwt_image, axis=0)

    age_pred, gender_pred = model.predict(dwt_image)
    age_range_class = np.argmax(age_pred)
    age_range = age_ranges[age_range_class]
    gender = 'Male' if gender_pred < 0.5 else 'Female'

    st.success(f"**Predicted Age Range:** {age_range}")
    st.success(f"**Predicted Gender:** {gender}")

# Expander for processed images
if (option == "Upload Fingerprint Image" and image is not None) or (option == "Use Fingerprint Scanner" and image is not None):
    with st.expander("🔍 View Processed Images"):
        st.image(denoised_image, caption="Denoised Image", use_column_width=True)
        st.image(edges_horizontal / np.max(edges_horizontal), caption="Horizontal Edges", use_column_width=True)
        st.image(edges_vertical / np.max(edges_vertical), caption="Vertical Edges", use_column_width=True)
        st.image(edges_diagonal / np.max(edges_diagonal), caption="Diagonal Edges", use_column_width=True)
import streamlit as st
import cv2
import numpy as np
import pywt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
import os

# Load the pre-trained models
model_path = 'age_gender_model(7).h5'
model_gender = load_model(model_path)

# Define the KNN model for age prediction
knn = KNeighborsClassifier(n_neighbors=5)

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

    dwt_features = np.concatenate([cA.flatten(), cH_thresh.flatten(), cV_thresh.flatten(), cD_thresh.flatten()])
    dwt_features = dwt_features / np.max(dwt_features)

    # Check the length of dwt_features
    print("Shape of dwt_features:", dwt_features.shape)

    dwt_image_size = 64 * 32  # Expected shape
    if dwt_features.size != dwt_image_size:
        raise ValueError(f"Expected {dwt_image_size} elements but got {dwt_features.size}.")

    dwt_image = dwt_features.reshape((64, 32))

    return denoised_image, cH_thresh, cV_thresh, cD_thresh, dwt_image

# Simulate capturing a fingerprint from a scanner
def capture_fingerprint_from_scanner():
    scanner_image_path = "scanner_fingerprint.bmp"  # This path would be dynamically created from the scanner
    if os.path.exists(scanner_image_path):
        image = cv2.imread(scanner_image_path, cv2.IMREAD_GRAYSCALE)
        return image
    else:
        st.error("Error: Scanner image not found!")
        return None

# Streamlit interface
st.set_page_config(page_title="Fingerprint Age and Gender Prediction", layout="wide", page_icon=":fingerprint:")

# Custom CSS styling
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        color: #333;
    }
    .stButton>button {
        background-color: #ff6600;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🔍 Fingerprint Age & Gender Prediction System")
st.subheader("Predict the **age range** and **gender** from fingerprint images or a live scanner.")

# Sidebar for user information
with st.sidebar:
    st.header("🔐 User Details")
    name = st.text_input("Name", help="Enter your full name.")
    id_number = st.text_input("ID Number", help="Enter your ID number.")
    st.write(f"👤 **User:** {name}")
    st.write(f"🆔 **ID Number:** {id_number}")

# Columns for layout
col1, col2 = st.columns(2)

# Initialize variables for handling unbound cases
image = None
denoised_image = None
edges_horizontal = None
edges_vertical = None
edges_diagonal = None

# Add options for uploading image or using scanner
option = st.radio("Choose input method:", ("Upload Fingerprint Image", "Use Fingerprint Scanner"))

with col1:
    if option == "Upload Fingerprint Image":
        uploaded_file = st.file_uploader("📁 Upload Fingerprint Image", type=["bmp", "jpg", "jpeg", "png"])
    else:
        st.write("🔌 Ensure your fingerprint scanner is connected.")
        if st.button("Capture Fingerprint from Scanner"):
            image = capture_fingerprint_from_scanner()
            if image is not None:
                st.image(image, caption="Captured Fingerprint Image", use_column_width=True)
            else:
                st.error("Unable to capture fingerprint.")

if option == "Upload Fingerprint Image" and uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image
    image_resized = cv2.resize(gray_image, (64, 32))
    
    # Apply DWT
    denoised_image, edges_horizontal, edges_vertical, edges_diagonal, dwt_image = apply_dwt(image_resized)

    dwt_image = np.expand_dims(dwt_image, axis=-1)
    dwt_image = np.expand_dims(dwt_image, axis=0)

    # Make predictions
    age_pred, gender_pred = model_gender.predict(dwt_image)
    
    # Age Prediction using KNN
    knn_flattened_data = dwt_image.reshape(dwt_image.shape[0], -1)  # Flatten for KNN
    age_pred_knn = knn.predict(knn_flattened_data)

    # Convert predictions
    age_range_class = np.argmax(age_pred)
    age_range = age_ranges[age_range_class]
    gender = 'Male' if gender_pred < 0.5 else 'Female'

    st.image(image, caption="Uploaded Fingerprint Image", use_column_width=True)
    st.success(f"**Predicted Age Range:** {age_range}")
    st.success(f"**Predicted Gender:** {gender}")

elif option == "Use Fingerprint Scanner" and image is not None:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised_image, edges_horizontal, edges_vertical, edges_diagonal, dwt_image = apply_dwt(gray_image)
    dwt_image = np.expand_dims(dwt_image, axis=-1)
    dwt_image = np.expand_dims(dwt_image, axis=0)

    # Make predictions
    age_pred, gender_pred = model_gender.predict(dwt_image)

    # Age Prediction using KNN
    knn_flattened_data = dwt_image.reshape(dwt_image.shape[0], -1)  # Flatten for KNN
    age_pred_knn = knn.predict(knn_flattened_data)

    # Convert predictions
    age_range_class = np.argmax(age_pred)
    age_range = age_ranges[age_range_class]
    gender = 'Male' if gender_pred < 0.5 else 'Female'

    st.success(f"**Predicted Age Range:** {age_range}")
    st.success(f"**Predicted Gender:** {gender}")

# Expander for processed images
if (option == "Upload Fingerprint Image" and image is not None) or (option == "Use Fingerprint Scanner" and image is not None):
    with st.expander("🔍 View Processed Images"):
        st.image(denoised_image, caption="Denoised Image", use_column_width=True)
        st.image(edges_horizontal / np.max(edges_horizontal), caption="Horizontal Edges", use_column_width=True)
        st.image(edges_vertical / np.max(edges_vertical), caption="Vertical Edges", use_column_width=True)
        st.image(edges_diagonal / np.max(edges_diagonal), caption="Diagonal Edges", use_column_width=True)
