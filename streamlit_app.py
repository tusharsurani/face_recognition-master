import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
from scipy.spatial import distance as dist

# Page configuration
st.set_page_config(
    page_title="Face Recognition App - All Features",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .stats-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .face-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load known faces
@st.cache_data
def load_known_faces():
    """Load known face encodings"""
    try:
        obama_image = cv2.imread("examples/obama.jpg")
        obama_face_encoding = DeepFace.represent(obama_image, model_name='VGG-Face', enforce_detection=False)[0]['embedding']
        
        biden_image = cv2.imread("examples/biden.jpg")
        biden_face_encoding = DeepFace.represent(biden_image, model_name='VGG-Face', enforce_detection=False)[0]['embedding']
        
        known_face_encodings = [
            obama_face_encoding,
            biden_face_encoding
        ]
        known_face_names = [
            "Barack Obama",
            "Joe Biden"
        ]
        
        return known_face_encodings, known_face_names
    except Exception as e:
        st.error(f"Error loading known faces: {str(e)}")
        return [], []

def process_image_recognition(image, known_face_encodings, known_face_names):
    """Process image for face detection and recognition"""
    image_array = np.array(image)
    frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Detect faces
    try:
        faces = DeepFace.extract_faces(frame, enforce_detection=False)
    except:
        faces = []
    
    face_locations = []
    face_encodings = []
    for face in faces:
        facial_area = face['facial_area']
        face_locations.append((facial_area['y'], facial_area['x'] + facial_area['w'], facial_area['y'] + facial_area['h'], facial_area['x']))  # top, right, bottom, left
        face_img = face['face']
        face_img = (face_img * 255).astype(np.uint8)
        try:
            encoding = DeepFace.represent(face_img, model_name='VGG-Face', enforce_detection=False)[0]['embedding']
            face_encodings.append(encoding)
        except:
            face_encodings.append(None)
    
    face_names = []
    for face_encoding in face_encodings:
        if face_encoding is None:
            face_names.append("Unknown")
            continue
        name = "Unknown"
        min_distance = float('inf')
        for known_encoding, known_name in zip(known_face_encodings, known_face_names):
            distance = np.linalg.norm(np.array(face_encoding) - np.array(known_encoding))
            if distance < 0.5:  # threshold
                if distance < min_distance:
                    min_distance = distance
                    name = known_name
        face_names.append(name)
    
    output_image = image_array.copy()
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(output_image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(output_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(output_image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    return output_image_rgb, face_names, len(face_locations)

def detect_faces(image, model="hog"):
    """Detect faces in image using HOG or CNN model"""
    image_array = np.array(image)
    frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    try:
        faces = DeepFace.extract_faces(frame, enforce_detection=False)
    except:
        faces = []
    
    face_locations = []
    for face in faces:
        facial_area = face['facial_area']
        face_locations.append((facial_area['y'], facial_area['x'] + facial_area['w'], facial_area['y'] + facial_area['h'], facial_area['x']))
    
    output_image = image_array.copy()
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    
    for top, right, bottom, left in face_locations:
        cv2.rectangle(output_image_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
    
    output_image_rgb = cv2.cvtColor(output_image_rgb, cv2.COLOR_BGR2RGB)
    return output_image_rgb, face_locations

def detect_facial_features(image):
    """Detect and draw facial landmarks - Note: DeepFace doesn't provide detailed landmarks like face_recognition"""
    st.warning("Facial landmarks detection is not available with the current setup. This feature requires detailed landmark detection which DeepFace doesn't provide.")
    return np.array(image), [], []

def calculate_face_distance(image1, image2):
    """Calculate distance between two faces"""
    img1_array = np.array(image1)
    img2_array = np.array(image2)
    
    try:
        result = DeepFace.verify(img1_array, img2_array, model_name='VGG-Face', enforce_detection=False)
        face_distance = result['distance']
        return face_distance, None
    except Exception as e:
        return None, str(e)

def detect_blink(image):
    """Detect if eyes are closed - Note: Blink detection requires facial landmarks which are not available"""
    st.warning("Blink detection is not available with the current setup. This feature requires detailed eye landmark detection.")
    return np.array(image), []

def get_ear(eye):
    """Calculate Eye Aspect Ratio"""
    if len(eye) < 6:
        return 0.0
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def apply_digital_makeup(image):
    """Apply digital makeup to faces - Note: Makeup requires facial landmarks"""
    st.warning("Digital makeup is not available with the current setup. This feature requires detailed facial landmark detection.")
    return np.array(image), 0

def blur_faces(image, blur_strength=30):
    """Blur faces in image"""
    image_array = np.array(image)
    frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    try:
        faces = DeepFace.extract_faces(frame, enforce_detection=False)
    except:
        faces = []
    
    for face in faces:
        facial_area = face['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        
        face_image = frame[y:y+h, x:x+w]
        face_image = cv2.GaussianBlur(face_image, (99, 99), blur_strength)
        frame[y:y+h, x:x+w] = face_image
    
    output_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return output_image, len(faces)

# Initialize session state
if 'processed_count' not in st.session_state:
    st.session_state.processed_count = 0
if 'total_faces' not in st.session_state:
    st.session_state.total_faces = 0
if 'recognition_history' not in st.session_state:
    st.session_state.recognition_history = []

# Load known faces
known_face_encodings, known_face_names = load_known_faces()

# Header
st.markdown('<div class="main-header">👤 Face Recognition App - All Features</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("Known Faces")
    for name in known_face_names:
        st.write(f"✅ {name}")
    
    st.divider()
    
    st.subheader("📊 Statistics")
    st.metric("Images Processed", st.session_state.processed_count)
    st.metric("Total Faces Detected", st.session_state.total_faces)
    
    if st.button("🔄 Reset Statistics", use_container_width=True):
        st.session_state.processed_count = 0
        st.session_state.total_faces = 0
        st.session_state.recognition_history = []
        st.rerun()
    
    st.divider()
    
    st.subheader("ℹ️ About")
    st.info("""
    **Available Features:**
    - ✅ Face Detection
    - ✅ Face Recognition
    - ❌ Facial Features Detection (requires detailed landmarks)
    - ✅ Face Distance Calculation
    - ❌ Blink Detection (requires eye landmarks)
    - ❌ Digital Makeup (requires facial landmarks)
    - ✅ Face Blurring
    
    Note: Some advanced features are not available in this cloud deployment due to library limitations. For full functionality, run the app locally with face_recognition library.
    
    Upload images or use webcam to try available features!
    """)

# Main content - Multiple tabs for different features
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🔍 Face Recognition", 
    "📷 Face Detection", 
    "🎭 Facial Features",
    "📏 Face Distance",
    "👁️ Blink Detection",
    "💄 Digital Makeup",
    "🔒 Face Blurring",
    "📜 History"
])

# Tab 1: Face Recognition
with tab1:
    st.header("Face Recognition")
    st.write("Recognize faces from known database (Barack Obama, Joe Biden)")
    
    col_input, col_webcam = st.columns(2)
    
    with col_input:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="recognition_upload")
    
    with col_webcam:
        camera_file = st.camera_input("Take Photo", key="recognition_camera")
    
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif camera_file is not None:
        image = Image.open(camera_file)
    
    if image is not None:
        with st.spinner("Processing..."):
            processed_image, face_names, face_count = process_image_recognition(
                image, known_face_encodings, known_face_names
            )
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📸 Original")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("🔍 Recognized")
            st.image(processed_image, use_container_width=True)
        
        if face_count > 0:
            st.success(f"✅ Detected {face_count} face(s)!")
            for i, name in enumerate(face_names, 1):
                st.write(f"**Face {i}:** {name}")
            
            st.session_state.processed_count += 1
            st.session_state.total_faces += face_count
            st.session_state.recognition_history.insert(0, {
                'type': 'Recognition',
                'faces': face_names,
                'count': face_count,
                'timestamp': st.session_state.processed_count
            })

# Tab 2: Face Detection
with tab2:
    st.header("Face Detection")
    st.write("Detect faces using HOG or CNN model")
    
    model_choice = st.radio("Detection Model", ["HOG (Faster)", "CNN (More Accurate)"], horizontal=True)
    model = "cnn" if model_choice == "CNN (More Accurate)" else "hog"
    
    col_input, col_webcam = st.columns(2)
    with col_input:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="detection_upload")
    with col_webcam:
        camera_file = st.camera_input("Take Photo", key="detection_camera")
    
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif camera_file is not None:
        image = Image.open(camera_file)
    
    if image is not None:
        with st.spinner("Detecting faces..."):
            processed_image, face_locations = detect_faces(image, model=model)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📸 Original")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("🔍 Detected Faces")
            st.image(processed_image, use_container_width=True)
        
        st.info(f"Found {len(face_locations)} face(s) using {model_choice} model")
        for i, (top, right, bottom, left) in enumerate(face_locations, 1):
            st.write(f"**Face {i}:** Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left}")

# Tab 3: Facial Features
with tab3:
    st.header("Facial Features Detection")
    st.write("Detect and visualize facial landmarks (eyes, nose, mouth, etc.)")
    
    st.warning("Facial landmarks detection is not available with the current cloud deployment setup. This feature requires detailed landmark detection which DeepFace doesn't provide. For full functionality, consider running the app locally or using a different deployment platform that supports face_recognition library.")

# Tab 4: Face Distance
with tab4:
    st.header("Face Distance Calculation")
    st.write("Compare similarity between two faces (lower distance = more similar)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Image 1")
        uploaded_file1 = st.file_uploader("Upload First Image", type=['jpg', 'jpeg', 'png'], key="distance1")
        camera_file1 = st.camera_input("Take Photo 1", key="distance_camera1")
    
    with col2:
        st.subheader("Image 2")
        uploaded_file2 = st.file_uploader("Upload Second Image", type=['jpg', 'jpeg', 'png'], key="distance2")
        camera_file2 = st.camera_input("Take Photo 2", key="distance_camera2")
    
    image1 = None
    image2 = None
    
    if uploaded_file1 is not None:
        image1 = Image.open(uploaded_file1)
    elif camera_file1 is not None:
        image1 = Image.open(camera_file1)
    
    if uploaded_file2 is not None:
        image2 = Image.open(uploaded_file2)
    elif camera_file2 is not None:
        image2 = Image.open(camera_file2)
    
    if image1 is not None and image2 is not None:
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(image1, use_container_width=True, caption="Image 1")
        with col_img2:
            st.image(image2, use_container_width=True, caption="Image 2")
        
        with st.spinner("Calculating face distance..."):
            distance, error = calculate_face_distance(image1, image2)
        
        if error:
            st.error(error)
        else:
            st.metric("Face Distance", f"{distance:.4f}")
            match_threshold_06 = distance < 0.6
            match_threshold_05 = distance < 0.5
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Match (0.6 threshold)", "✅ Yes" if match_threshold_06 else "❌ No")
            with col2:
                st.metric("Match (0.5 threshold)", "✅ Yes" if match_threshold_05 else "❌ No")
            with col3:
                similarity = max(0, (1 - distance) * 100)
                st.metric("Similarity", f"{similarity:.1f}%")

# Tab 5: Blink Detection
with tab5:
    st.header("Blink Detection")
    st.write("Detect if eyes are open or closed using Eye Aspect Ratio (EAR)")
    
    st.warning("Blink detection is not available with the current cloud deployment setup. This feature requires detailed eye landmark detection which DeepFace doesn't provide. For full functionality, consider running the app locally or using a different deployment platform that supports face_recognition library.")

# Tab 6: Digital Makeup
with tab6:
    st.header("Digital Makeup")
    st.write("Apply digital makeup to faces (eyebrows, lipstick, eyeliner, eye sparkle)")
    
    st.warning("Digital makeup is not available with the current cloud deployment setup. This feature requires detailed facial landmark detection which DeepFace doesn't provide. For full functionality, consider running the app locally or using a different deployment platform that supports face_recognition library.")

# Tab 7: Face Blurring
with tab7:
    st.header("Face Blurring (Privacy)")
    st.write("Blur faces in images for privacy protection")
    
    blur_strength = st.slider("Blur Strength", 10, 50, 30, 5)
    
    col_input, col_webcam = st.columns(2)
    with col_input:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="blur_upload")
    with col_webcam:
        camera_file = st.camera_input("Take Photo", key="blur_camera")
    
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif camera_file is not None:
        image = Image.open(camera_file)
    
    if image is not None:
        with st.spinner("Blurring faces..."):
            processed_image, face_count = blur_faces(image, blur_strength=blur_strength)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📸 Original")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("🔒 Blurred")
            st.image(processed_image, use_container_width=True)
        
        st.info(f"✅ Blurred {face_count} face(s) for privacy")

# Tab 8: History
with tab8:
    st.header("Recognition History")
    
    if st.session_state.recognition_history:
        st.write(f"Total records: {len(st.session_state.recognition_history)}")
        
        for idx, record in enumerate(st.session_state.recognition_history[:20], 1):
            with st.expander(f"Record #{record['timestamp']} - {record['type']} - {record['count']} face(s)"):
                st.write(f"**Type:** {record['type']}")
                if 'faces' in record:
                    st.write("**Detected Faces:**")
                    for i, face_name in enumerate(record['faces'], 1):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"{i}. {face_name}")
                        with col2:
                            if face_name != "Unknown":
                                st.success("✅ Known")
                            else:
                                st.info("❓ Unknown")
    else:
        st.info("No recognition history yet. Start using the features to see history here.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Face Recognition App with All Features - Powered by Streamlit & face_recognition library</p>
</div>
""", unsafe_allow_html=True)
