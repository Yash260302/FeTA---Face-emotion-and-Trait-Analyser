import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from camera import detect_faces
from emotion_detection import EmotionDetector

st.set_page_config(page_title="Age, Gender & Emotion Detector - Photo Mode", layout="wide")
st.title("📸 Age, Gender & Emotion Detector - Photo Mode")

# IMPORTANT: Must match training normalization
MAX_AGE = 116.0

# Cache models so they load once
@st.cache_resource
def load_age_gender_model():
    """Try to load improved model first, fallback to original"""
    models_to_try = [
        ("age_gender_model_improved.keras", (128, 128)),
        ("age_gender_model.keras", (64, 64))
    ]
    
    for model_path, img_size in models_to_try:
        try:
            m = load_model(model_path, compile=False)
            st.sidebar.success(f"✓ Loaded: {model_path}")
            return m, img_size
        except Exception as e:
            st.sidebar.warning(f"Failed to load {model_path}: {e}")
            continue
    
    raise Exception("No age/gender model found!")

@st.cache_resource
def load_emotion_detector():
    return EmotionDetector(model_path="models/best_emotion_model.h5")

# Load models
try:
    age_gender_model, MODEL_IMAGE_SIZE = load_age_gender_model()
    emotion_detector = load_emotion_detector()
    gender_labels = ["Male", "Female"]
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False
    MODEL_IMAGE_SIZE = (64, 64)

def preprocess_for_age_gender(face_bgr, target_size=MODEL_IMAGE_SIZE):
    """BGR -> RGB, resize, normalize to [0,1]"""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, target_size)
    arr = img_to_array(face_resized).astype("float32") / 255.0
    return arr

def process_image(image):
    """Process uploaded image and detect faces with predictions"""
    # Convert PIL to BGR for OpenCV
    img_array = np.array(image)
    if len(img_array.shape) == 2:  # Grayscale
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:  # RGBA
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:  # RGB
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Detect faces with better parameters
    boxes = detect_faces(img_bgr, scaleFactor=1.05, minNeighbors=6, minSize=(50, 50))
    
    if len(boxes) == 0:
        return None, "No faces detected in the image. Please try another photo with a clearer face."
    
    results = []
    
    # Process each face
    for i, (x, y, w, h) in enumerate(boxes):
        # Add more padding for better face capture
        pad_x = int(0.2 * w)
        pad_y = int(0.2 * h)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_bgr.shape[1], x + w + pad_x)
        y2 = min(img_bgr.shape[0], y + h + pad_y)
        
        face_crop = img_bgr[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            continue
        
        # Age and Gender prediction
        face_processed = preprocess_for_age_gender(face_crop)
        face_batch = np.expand_dims(face_processed, axis=0)
        
        try:
            preds = age_gender_model.predict(face_batch, verbose=0)
            
            # Extract predictions
            if isinstance(preds, (list, tuple)) and len(preds) == 2:
                age_pred, gender_pred = preds
            else:
                age_pred = preds[:, 0:1]
                gender_pred = preds[:, 1:]
            
            # Process age with better rounding
            age_norm = float(age_pred[0].ravel()[0])
            age_years = int(round(age_norm * MAX_AGE))
            age_years = max(0, min(116, age_years))
            
            # Process gender
            gender_probs = np.asarray(gender_pred[0])
            gender_idx = int(np.argmax(gender_probs))
            gender_confidence = float(gender_probs[gender_idx])
            
            if gender_confidence > 0.5:
                gender = gender_labels[gender_idx]
            else:
                gender = "Unknown"
            
            # Emotion prediction
            if emotion_detector.is_ready():
                emo_labels, emo_confs = emotion_detector.predict_emotions_for_faces([face_crop])
                emotion = emo_labels[0]
                emotion_conf = emo_confs[0]
            else:
                emotion = "Unknown"
                emotion_conf = 0.0
            
            # Draw on image with thicker lines
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 4)
            
            # Prepare label
            label = f"{gender}, {age_years}y | {emotion}"
            
            # Draw label background with better visibility
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            cv2.rectangle(
                img_bgr,
                (x, y - text_height - 15),
                (x + text_width + 10, y),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                img_bgr,
                label,
                (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )
            
            results.append({
                "face_num": i + 1,
                "gender": gender,
                "gender_confidence": gender_confidence,
                "age": age_years,
                "emotion": emotion,
                "emotion_confidence": emotion_conf,
                "bbox": (x, y, w, h)
            })
            
        except Exception as e:
            st.error(f"Error processing face {i+1}: {e}")
            continue
    
    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return img_rgb, results

# Main UI
st.markdown("""
### 📋 How to use:
1. **Upload** a photo or **take one** with your webcam
2. Click **"Analyze Photo"** to detect faces and get predictions
3. View results with bounding boxes and detailed information

💡 **Tip**: Use good lighting and face the camera directly for best results!
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Photo")
    upload_option = st.radio(
        "Choose input method:",
        ["📁 Upload from file", "📷 Take photo with webcam"],
        horizontal=False
    )
    
    uploaded_image = None
    
    if upload_option == "📁 Upload from file":
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a photo containing faces"
        )
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    
    else:  # Webcam option
        st.info("📸 Position yourself and click the camera button below")
        camera_photo = st.camera_input("Take a photo")
        if camera_photo is not None:
            uploaded_image = Image.open(camera_photo)
            st.image(uploaded_image, caption="Captured Photo", use_container_width=True)

with col2:
    st.subheader("📊 Analysis Results")
    
    if uploaded_image is not None and models_loaded:
        if st.button("🔍 Analyze Photo", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing image... Please wait..."):
                result_image, results = process_image(uploaded_image)
                
                if result_image is not None:
                    st.image(result_image, caption="Detected Faces with Predictions", use_container_width=True)
                    
                    st.success(f"✅ Successfully detected {len(results)} face(s)")
                    
                    # Display results in organized cards
                    for result in results:
                        with st.container():
                            # Color code by confidence
                            bg_color = "#e8f5e9" if result['gender_confidence'] > 0.7 else "#fff9c4"
                            
                            st.markdown(f"""
                            <div style="background-color: {bg_color}; padding: 20px; border-radius: 12px; margin: 15px 0; border-left: 5px solid #4CAF50;">
                                <h3 style="margin-top: 0; color: #2e7d32;">👤 Face #{result['face_num']}</h3>
                                <div style="font-size: 16px; line-height: 1.8;">
                                    <p><strong>👔 Gender:</strong> {result['gender']} 
                                       <span style="color: #666;">({result['gender_confidence']:.1%} confidence)</span></p>
                                    <p><strong>🎂 Age:</strong> {result['age']} years old</p>
                                    <p><strong>😊 Emotion:</strong> {result['emotion']} 
                                       <span style="color: #666;">({result['emotion_confidence']:.1%} confidence)</span></p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Add interpretation
                            if result['age'] < 18:
                                st.info(f"👶 Appears to be a minor (under 18)")
                            elif result['age'] < 30:
                                st.info(f"👨 Young adult")
                            elif result['age'] < 60:
                                st.info(f"👨‍💼 Middle-aged adult")
                            else:
                                st.info(f"👴 Senior")
                else:
                    st.warning(results)  # This will be the error message
    elif uploaded_image is not None and not models_loaded:
        st.error("❌ Models failed to load. Please check if model files exist in the correct locations.")
    else:
        st.info("👆 Upload or capture a photo above to begin analysis")

# Sidebar with tips and information
st.sidebar.header("💡 Tips for Best Results")
st.sidebar.markdown("""
**For accurate predictions:**
- ✅ **Good lighting**: Natural daylight works best
- ✅ **Face the camera**: Direct frontal view (not profile)
- ✅ **Distance**: Stay 1-2 meters from camera
- ✅ **Clear image**: Avoid blur, sunglasses, or masks
- ✅ **Single person**: One face at a time is more accurate
- ✅ **Neutral background**: Plain backgrounds reduce errors
- ✅ **No extreme angles**: Keep head upright

**What affects accuracy:**
- ❌ Poor lighting or shadows
- ❌ Profile views or tilted heads
- ❌ Accessories (hats, sunglasses, masks)
- ❌ Heavy makeup or filters
- ❌ Very young children (under 5)
- ❌ Extreme ages (over 80)
""")

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
### 🤖 Model Information
- **Image Size**: {MODEL_IMAGE_SIZE[0]}x{MODEL_IMAGE_SIZE[1]}
- **Age Range**: 0-116 years
- **Gender**: Binary (Male/Female)
- **Emotions**: 7 categories
- **Architecture**: ResNet-inspired CNN

### 📊 Expected Performance
- **Age Accuracy**: ±5-8 years
- **Gender Accuracy**: 85-95%
- **Emotion Accuracy**: 65-75%

*Note: Results may vary based on image quality and subject characteristics*
""")

st.sidebar.markdown("---")
st.sidebar.caption("Version 2.0 - Improved Model")