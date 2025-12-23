import streamlit as st
import face_recognition
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
        obama_image = face_recognition.load_image_file("examples/obama.jpg")
        obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
        
        biden_image = face_recognition.load_image_file("examples/biden.jpg")
        biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
        
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
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if best_match_index < len(matches) and matches[best_match_index]:
                name = known_face_names[best_match_index]
        
        face_names.append(name)
    
    output_image = image_array.copy()
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        cv2.rectangle(output_image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(output_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(output_image, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
    
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    return output_image_rgb, face_names, len(face_locations)

def detect_faces(image, model="hog"):
    """Detect faces in image using HOG or CNN model"""
    image_array = np.array(image)
    face_locations = face_recognition.face_locations(image_array, model=model)
    
    output_image = image_array.copy()
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR) if len(output_image.shape) == 3 else output_image
    
    for top, right, bottom, left in face_locations:
        cv2.rectangle(output_image_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
    
    output_image_rgb = cv2.cvtColor(output_image_rgb, cv2.COLOR_BGR2RGB) if len(output_image_rgb.shape) == 3 else output_image_rgb
    return output_image_rgb, face_locations

def detect_facial_features(image):
    """Detect and draw facial landmarks"""
    image_array = np.array(image)
    face_landmarks_list = face_recognition.face_landmarks(image_array)
    
    pil_image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(pil_image)
    
    features_info = []
    for face_idx, face_landmarks in enumerate(face_landmarks_list):
        face_info = {"face": face_idx + 1, "features": {}}
        
        for facial_feature in face_landmarks.keys():
            points = face_landmarks[facial_feature]
            draw.line(points, fill=(255, 0, 0), width=3)
            face_info["features"][facial_feature] = len(points)
        
        features_info.append(face_info)
    
    return np.array(pil_image), face_landmarks_list, features_info

def calculate_face_distance(image1, image2):
    """Calculate distance between two faces"""
    img1_array = np.array(image1)
    img2_array = np.array(image2)
    
    encodings1 = face_recognition.face_encodings(img1_array)
    encodings2 = face_recognition.face_encodings(img2_array)
    
    if len(encodings1) == 0:
        return None, "No face found in first image"
    if len(encodings2) == 0:
        return None, "No face found in second image"
    
    face_distance = face_recognition.face_distance([encodings1[0]], encodings2[0])[0]
    return face_distance, None

def detect_blink(image):
    """Detect if eyes are closed"""
    image_array = np.array(image)
    face_landmarks_list = face_recognition.face_landmarks(image_array)
    
    blink_results = []
    output_image = image_array.copy()
    
    pil_image = Image.fromarray(output_image)
    draw = ImageDraw.Draw(pil_image)
    
    for face_idx, face_landmarks in enumerate(face_landmarks_list):
        left_eye = face_landmarks.get('left_eye', [])
        right_eye = face_landmarks.get('right_eye', [])
        
        if len(left_eye) >= 6 and len(right_eye) >= 6:
            ear_left = get_ear(left_eye)
            ear_right = get_ear(right_eye)
            avg_ear = (ear_left + ear_right) / 2.0
            
            is_closed = avg_ear < 0.2
            blink_results.append({
                "face": face_idx + 1,
                "left_ear": ear_left,
                "right_ear": ear_right,
                "avg_ear": avg_ear,
                "is_closed": is_closed
            })
            
            # Draw eye regions
            eye_color = (255, 0, 0) if is_closed else (0, 255, 0)
            for point in left_eye + right_eye:
                draw.ellipse([point[0]-3, point[1]-3, point[0]+3, point[1]+3], fill=eye_color)
    
    output_image = np.array(pil_image)
    
    return output_image, blink_results

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
    """Apply digital makeup to faces"""
    image_array = np.array(image)
    face_landmarks_list = face_recognition.face_landmarks(image_array)
    
    pil_image = Image.fromarray(image_array)
    
    for face_landmarks in face_landmarks_list:
        draw = ImageDraw.Draw(pil_image, 'RGBA')
        
        # Make the eyebrows
        if 'left_eyebrow' in face_landmarks:
            draw.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
            draw.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
        if 'right_eyebrow' in face_landmarks:
            draw.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
            draw.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)
        
        # Gloss the lips
        if 'top_lip' in face_landmarks:
            draw.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
            draw.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
        if 'bottom_lip' in face_landmarks:
            draw.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
            draw.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)
        
        # Sparkle the eyes
        if 'left_eye' in face_landmarks:
            draw.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
            draw.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
        if 'right_eye' in face_landmarks:
            draw.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))
            draw.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)
    
    return np.array(pil_image), len(face_landmarks_list)

def blur_faces(image, blur_strength=30):
    """Blur faces in image"""
    image_array = np.array(image)
    frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    for top, right, bottom, left in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        face_image = frame[top:bottom, left:right]
        face_image = cv2.GaussianBlur(face_image, (99, 99), blur_strength)
        frame[top:bottom, left:right] = face_image
    
    output_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return output_image, len(face_locations)

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
    - Face Detection (HOG & CNN)
    - Face Recognition
    - Facial Features Detection
    - Face Distance Calculation
    - Blink Detection
    - Digital Makeup
    - Face Blurring
    
    Upload images or use webcam to try all features!
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
    
    col_input, col_webcam = st.columns(2)
    with col_input:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="features_upload")
    with col_webcam:
        camera_file = st.camera_input("Take Photo", key="features_camera")
    
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif camera_file is not None:
        image = Image.open(camera_file)
    
    if image is not None:
        with st.spinner("Detecting facial features..."):
            processed_image, landmarks_list, features_info = detect_facial_features(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📸 Original")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("🎭 Features Detected")
            st.image(processed_image, use_container_width=True)
        
        st.success(f"✅ Detected {len(landmarks_list)} face(s) with facial features!")
        for info in features_info:
            with st.expander(f"Face {info['face']} - Facial Features"):
                for feature, point_count in info['features'].items():
                    st.write(f"**{feature.replace('_', ' ').title()}:** {point_count} points")

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
    
    col_input, col_webcam = st.columns(2)
    with col_input:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="blink_upload")
    with col_webcam:
        camera_file = st.camera_input("Take Photo", key="blink_camera")
    
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif camera_file is not None:
        image = Image.open(camera_file)
    
    if image is not None:
        with st.spinner("Detecting blinks..."):
            processed_image, blink_results = detect_blink(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📸 Original")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("👁️ Blink Detection")
            st.image(processed_image, use_container_width=True)
        
        if blink_results:
            for result in blink_results:
                with st.expander(f"Face {result['face']} - Eye Status"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Left EAR", f"{result['left_ear']:.4f}")
                        st.metric("Right EAR", f"{result['right_ear']:.4f}")
                        st.metric("Average EAR", f"{result['avg_ear']:.4f}")
                    with col2:
                        status = "👁️ Closed" if result['is_closed'] else "👀 Open"
                        st.metric("Eye Status", status)
                        st.info("EAR < 0.2 indicates closed eyes")

# Tab 6: Digital Makeup
with tab6:
    st.header("Digital Makeup")
    st.write("Apply digital makeup to faces (eyebrows, lipstick, eyeliner, eye sparkle)")
    
    col_input, col_webcam = st.columns(2)
    with col_input:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="makeup_upload")
    with col_webcam:
        camera_file = st.camera_input("Take Photo", key="makeup_camera")
    
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif camera_file is not None:
        image = Image.open(camera_file)
    
    if image is not None:
        with st.spinner("Applying digital makeup..."):
            processed_image, face_count = apply_digital_makeup(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📸 Original")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("💄 With Makeup")
            st.image(processed_image, use_container_width=True)
        
        st.success(f"✅ Applied makeup to {face_count} face(s)!")

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
