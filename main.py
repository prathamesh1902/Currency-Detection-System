import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from gtts import gTTS
import tempfile
import base64

# Configure the page
st.set_page_config(page_title="Indian Currency Classifier", layout="centered")
st.title("Indian Currency Classifier with Auto-Voice Note")

# Image dimensions (must match your training)
IMG_SIZE = (224, 224)

def load_trained_model():
    st.info("Loading model...")
    try:
        model = tf.keras.models.load_model('final_currency_models.keras')
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load your trained model
model = load_trained_model()

# Define your class names exactly as used during training.
class_names = ['10', '100', '20', '200', '2000', '50', '500', 'Background']

# Language selection
st.write("### Select Language for Voice Note")
language_options = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr"
}
selected_language = st.selectbox("Choose a language", list(language_options.keys()))
language_code = language_options[selected_language]

def preprocess_image(img: Image.Image):
    try:
        img = img.resize(IMG_SIZE)  # Resize to expected dimensions
        img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def generate_voice_note(text: str, lang: str):
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            with open(fp.name, "rb") as f:
                audio_bytes = f.read()
        return audio_bytes
    except Exception as e:
        st.error(f"Error generating voice note: {e}")
        return None

# Option to capture image from camera or upload from computer
st.write("### Upload or Capture an Image")
uploaded_file = st.file_uploader("Upload an image of an Indian currency note", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("Capture an image using your camera")

# Process the selected image
img = None
if uploaded_file is not None:
    img = Image.open(uploaded_file)
elif camera_file is not None:
    img = Image.open(camera_file)

if img is not None:
    try:
        st.image(img, caption="Selected Image", use_column_width=True)
        
        # Preprocess the image
        processed_img = preprocess_image(img)
        
        if processed_img is not None and model is not None:
            st.info("Running prediction...")
            prediction = model.predict(processed_img)
            pred_index = np.argmax(prediction)
            pred_class = class_names[pred_index]
            st.write("### Predicted Currency:")
            st.write(pred_class)
            
            # Text for voice note based on language
            if selected_language == "English":
                voice_text = f"The predicted currency denomination is {pred_class}"
            elif selected_language == "Hindi":
                voice_text = f"अनुमानित मुद्रा मूल्य {pred_class} है"
            elif selected_language == "Marathi":
                voice_text = f"अंदाजे चलन मूल्य {pred_class} आहे"
            
            # Generate and play the voice note
            audio_data = generate_voice_note(voice_text, language_code)
            if audio_data:
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                audio_html = f"""
                <audio controls autoplay>
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
