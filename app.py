import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
import io

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = tf.compat.v1.layers.flatten(y_true)
    y_pred_f = tf.compat.v1.layers.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def load_and_preprocess_image(uploaded_file):
    content = uploaded_file.read()
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return image

def make_prediction(model, image):
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

# Load your ResUNet model
model_path = "ResUNet_100ep.h5"
model = keras.models.load_model(model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef}, compile=False)

# Streamlit app

# Markdown text for the description
description_text = """
## ResUNet: Residual U-Net for Image Segmentation

ResUNet is a convolutional neural network architecture designed for semantic segmentation tasks. 
It extends the U-Net architecture by incorporating residual connections for improved training and performance.

**Key Features:**
- U-Net architecture with encoder and decoder paths.
- Residual connections for improved optimization.
- Batch normalization and activation functions.
- Dropout for preventing overfitting.
- Dice coefficient loss for segmentation tasks.

"""

st.title("Salt Segmentation with ResUNet")
# st.write("ResUNet (Residual U-Net) is a convolutional neural network (CNN) architecture designed for semantic segmentation tasks, where the goal is to classify each pixel in an image into different classes. ResUNet is an extension of the U-Net architecture, incorporating residual connections to enhance the training and performance of the network. ")
st.markdown(description_text, unsafe_allow_html=True)
st.sidebar.image("resunet_architecture.png", caption="ResUNet Architecture", use_column_width=True)

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    input_image = load_and_preprocess_image(uploaded_file)

    # Make prediction
    prediction = make_prediction(model, input_image)

    # Display the original and segmented images side by side
    st.subheader("Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(input_image, caption="Original Image", use_column_width=True, channels="GRAY")

    with col2:
        st.image(np.squeeze(prediction), caption="Segmentation Result", use_column_width=True, channels="GRAY")
