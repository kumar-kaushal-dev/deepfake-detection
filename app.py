import streamlit as st
import os
import tempfile
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import time
from face_utils import FaceDetector
from model import DeepfakeClassifier, Preprocessor, load_model, HuggingFaceDeepfakeDetector
from video_processor import VideoProcessor

# Import advanced model if available
try:
    from advanced_model import DeepfakeDetector, VideoDeepfakeDetector
    ADVANCED_MODEL_AVAILABLE = True
except ImportError:
    try:
        # Try to import the fallback model if advanced model is not available
        from fallback_model import FallbackDeepfakeDetector, FallbackVideoDetector
        ADVANCED_MODEL_AVAILABLE = True  # We can still use the fallback as "advanced"
    except ImportError:
        ADVANCED_MODEL_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
PRETRAINED_DIR = "pretrained_models"
TEMP_DIR = tempfile.gettempdir()
UPLOAD_DIR = "uploaded_files"
HUGGINGFACE_MODEL_DIR = os.path.join(PRETRAINED_DIR, "Deep-Fake-Detector-Model")
ADVANCED_MODEL_PATH = os.path.join(PRETRAINED_DIR, "efficientnet_b4_deepfake.pth")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PRETRAINED_DIR, exist_ok=True)

# Function to get the best available model
def get_model_path(model_name):
    """
    Get the path to the best available model for the selected architecture
    
    Args:
        model_name (str): Name of the model architecture
        
    Returns:
        str: Path to the model file
    """
    # Check for advanced model
    if model_name == 'advanced_efficientnet_b4':
        if os.path.exists(ADVANCED_MODEL_PATH):
            return ADVANCED_MODEL_PATH
        else:
            st.warning("""
            Advanced EfficientNet-B4 model not found. Please run the download_advanced_model.py script first:
            ```
            python download_advanced_model.py
            ```
            Falling back to HuggingFace ViT model.
            """)
            # Fall back to HuggingFace model
            model_name = 'huggingface_vit'
    
    # Check for HuggingFace model
    if model_name == 'huggingface_vit':
        if os.path.exists(HUGGINGFACE_MODEL_DIR):
            return HUGGINGFACE_MODEL_DIR
        else:
            st.warning("""
            HuggingFace model not found. Please run the download_model.py script first:
            ```
            python download_model.py
            ```
            Falling back to EfficientNet model.
            """)
            # Fall back to EfficientNet
            model_name = 'efficientnet_b0'
    
    # Mapping of model names to potential pretrained files
    model_files = {
        'efficientnet_b0': [
            os.path.join(PRETRAINED_DIR, "efficientnet_b0_deepfake.pth"),
            os.path.join(PRETRAINED_DIR, "efficientnet_c40.pth"),
            os.path.join(PRETRAINED_DIR, "efficientnet_celebdf.pth"),
            "deepfake_model.pth"  # Fallback to default
        ],
        'efficientnet_b3': [
            os.path.join(PRETRAINED_DIR, "efficientnet_b3_deepfake.pth"),
            os.path.join(PRETRAINED_DIR, "efficientnet_c40.pth"),
            os.path.join(PRETRAINED_DIR, "efficientnet_celebdf.pth"),
            "deepfake_model.pth"  # Fallback to default
        ],
        'xception': [
            os.path.join(PRETRAINED_DIR, "xception_c40.pth"),
            os.path.join(PRETRAINED_DIR, "xception_celebdf.pth")
        ]
    }
    
    # Get the list of potential files for the model
    potential_files = model_files.get(model_name, [])
    
    # Find the first existing file
    for file_path in potential_files:
        if os.path.exists(file_path):
            return file_path
    
    # If no model file found, use default
    return None

# Load advanced model
def load_advanced_model(model_path, device='cpu'):
    """
    Load the advanced deepfake detection model
    
    Args:
        model_path (str): Path to the model file
        device (str): Device to run the model on
        
    Returns:
        VideoDeepfakeDetector or FallbackVideoDetector: Model for deepfake detection
    """
    try:
        # First try to load using the advanced model
        from advanced_model import VideoDeepfakeDetector
        # Use VideoDeepfakeDetector which wraps the model
        advanced_detector = VideoDeepfakeDetector(
            model_path=model_path,
            model_name='efficientnet_b4',
            device=device
        )
        print("Using advanced EfficientNet-B4 model")
        return advanced_detector
    except ImportError:
        # If advanced model not available, try using the fallback
        try:
            from fallback_model import FallbackVideoDetector
            # Use FallbackVideoDetector
            fallback_detector = FallbackVideoDetector(
                model_name='efficientnet_b4',
                device=device
            )
            print("Using fallback EfficientNet model")
            return fallback_detector
        except ImportError:
            print("No deepfake detection models available")
            return None

# Function to load model and initialize processors
@st.cache_resource
def load_detection_model(model_name="efficientnet_b0"):
    """
    Load the deepfake detection model and initialize processors
    
    Args:
        model_name (str): Name of the model architecture
        
    Returns:
        tuple: (face_detector, preprocessor, model, device, is_advanced_model)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if using advanced model
    if model_name == 'advanced_efficientnet_b4' and ADVANCED_MODEL_AVAILABLE:
        model_path = get_model_path(model_name)
        if model_path and os.path.exists(model_path):
            # Load advanced model
            advanced_detector = load_advanced_model(model_path, device)
            if advanced_detector:
                # Still need face detector for other functions
                face_detector = FaceDetector(device=device)
                # Use a dummy preprocessor (not used with advanced model)
                preprocessor = Preprocessor(image_size=(224, 224))
                return face_detector, preprocessor, advanced_detector, device, True
    
    # If not using advanced model or it failed to load, use standard models
    # Initialize face detector
    face_detector = FaceDetector(device=device)
    
    # Initialize preprocessor
    preprocessor = Preprocessor(image_size=(224, 224))
    
    # Find the best available model path
    model_path = get_model_path(model_name)
    
    # Load model
    model = load_model(model_path=model_path, model_name=model_name, device=device)
    
    return face_detector, preprocessor, model, device, False

# Function to display video frames in a grid
def display_video_frames(frames, columns=3):
    """
    Display video frames in a grid layout
    
    Args:
        frames (list): List of video frames
        columns (int): Number of columns in the grid
    """
    # Calculate number of rows needed
    rows = len(frames) // columns + (1 if len(frames) % columns > 0 else 0)
    
    # Create a grid of frames
    for row in range(rows):
        cols = st.columns(columns)
        for col in range(columns):
            idx = row * columns + col
            if idx < len(frames):
                # Convert frame to RGB for display
                rgb_frame = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
                cols[col].image(rgb_frame, use_container_width=True, caption=f"Frame {idx+1}")

# Function to display verdict
def display_verdict(fake_probabilities):
    """
    Display a clear verdict about whether content is fake or real
    
    Args:
        fake_probabilities (list): List of fake probabilities
    """
    if not fake_probabilities:
        return
    
    # Calculate average probability
    avg_prob = sum(fake_probabilities) / len(fake_probabilities)
    max_prob = max(fake_probabilities)
    
    # Determine if fake based on probability (0.5 is fixed internally)
    is_fake = max_prob >= 0.5
    
    # Display verdict
    st.markdown("---")
    st.header("VERDICT")
    
    # Create columns for layout
    col1, col2 = st.columns([1, 4])
    
    with col1:
        # Display icon
        if is_fake:
            st.markdown("### 🔴")
        else:
            st.markdown("### 🟢")
    
    with col2:
        # Display verdict text with appropriate color
        if is_fake:
            st.markdown(f'<h2 style="color: red;">FAKE DETECTED</h2>', unsafe_allow_html=True)
            st.markdown(f'<p>Max fake probability: <b>{max_prob:.2f}</b></p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<h2 style="color: green;">LIKELY REAL</h2>', unsafe_allow_html=True)
            st.markdown(f'<p>Max fake probability: <b>{max_prob:.2f}</b></p>', unsafe_allow_html=True)
        
        # Display confidence level
        confidence = abs(max_prob - 0.5) * 2  # Scale to 0-1
        st.progress(confidence, text=f"Confidence: {confidence:.2%}")

# Function to process an uploaded image
def process_image(image_file, face_detector, preprocessor, model, device='cpu'):
    """
    Process an uploaded image for deepfake detection
    
    Args:
        image_file: Uploaded image file
        face_detector: Face detector
        preprocessor: Image preprocessor
        model: Deepfake detection model
        device: Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        tuple: (result_image, detection_results)
    """
    # Read the image
    image = Image.open(image_file).convert('RGB')
    image_np = np.array(image)
    
    # Convert RGB to BGR (OpenCV format)
    bgr_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Detect faces
    faces, boxes = face_detector.detect_faces(bgr_image)
    
    # If no faces detected
    if not faces:
        st.warning("No faces detected in the image.")
        return image_np, {"faces_detected": 0, "fake_probabilities": []}
    
    # Preprocess faces
    face_tensors = preprocessor.preprocess_batch(faces)
    face_tensors = face_tensors.to(device)  # Use device directly
    
    # Make predictions
    with torch.no_grad():
        predictions = model(face_tensors).cpu().numpy().flatten()
    
    # Draw results on the image
    result_image = bgr_image.copy()
    
    for i, (box, pred) in enumerate(zip(boxes, predictions)):
        x1, y1, x2, y2 = box
        
        # Determine color based on prediction (red for fake, green for real)
        # Higher value = more likely to be fake
        color = (0, int(255 * (1 - pred)), int(255 * pred))  # BGR format
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        label = f"Fake: {pred:.2f}"
        
        # Draw label background
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result_image, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), 
                     color, 
                     cv2.FILLED)
        
        # Draw label text
        cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Convert BGR back to RGB for display
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    # Prepare detection results
    detection_results = {
        "faces_detected": len(faces),
        "fake_probabilities": predictions.tolist()
    }
    
    return result_rgb, detection_results

# Function to show prediction results
def show_prediction_results(detection_results):
    """
    Display prediction results with metrics and visualizations
    
    Args:
        detection_results (dict): Detection results
    """
    if detection_results["faces_detected"] == 0:
        st.warning("No faces were detected in the image/video.")
        return
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    # Number of faces detected
    col1.metric("Faces Detected", detection_results["faces_detected"])
    
    # Calculate statistics
    if detection_results["fake_probabilities"]:
        avg_prob = sum(detection_results["fake_probabilities"]) / len(detection_results["fake_probabilities"])
        max_prob = max(detection_results["fake_probabilities"])
        
        col2.metric("Average Fake Probability", f"{avg_prob:.2f}")
        col3.metric("Max Fake Probability", f"{max_prob:.2f}")
        
        # Plot probability distribution if multiple faces
        if len(detection_results["fake_probabilities"]) > 1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(detection_results["fake_probabilities"], bins=10, alpha=0.7, color='blue')
            # Use fixed 0.5 threshold line
            ax.axvline(x=0.5, color='red', linestyle='--', label=f'Threshold (0.5)')
            ax.set_xlabel('Fake Probability')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Fake Probabilities')
            ax.legend()
            st.pyplot(fig)

# Function to process an uploaded image with the advanced model
def process_image_advanced(image_file, detector):
    """
    Process an uploaded image with the advanced deepfake detector
    
    Args:
        image_file: Uploaded image file
        detector: Advanced deepfake detector
        
    Returns:
        tuple: (result_image, detection_results)
    """
    # Read the image
    image = Image.open(image_file).convert('RGB')
    image_np = np.array(image)
    
    # Process the image
    prob = detector.detect_from_frame(image_np)
    
    # Create result image with prediction
    result_image = image_np.copy()
    
    # Add text showing prediction
    label = f"Deepfake: {prob:.2f}"
    color = (0, 0, 255) if prob > 0.5 else (0, 255, 0)  # BGR format
    cv2.putText(result_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Create detection results
    detection_results = {
        "faces_detected": 1,  # Simplified for full-image analysis
        "fake_probabilities": [prob]
    }
    
    return result_image, detection_results

def main():
    # Add title and description
    st.title("🕵️‍♀️ Deepfake Detection App")
    st.markdown("""
    This application detects deepfake content in images and videos using pretrained deep learning models.
    Upload an image or video file to get started.
    """)
    
    # Check if any pretrained models are available
    no_models = len([f for f in os.listdir(PRETRAINED_DIR) if f.endswith('.pth') or os.path.isdir(os.path.join(PRETRAINED_DIR, f))]) == 0 if os.path.exists(PRETRAINED_DIR) else True
    
    if no_models:
        st.warning("""
        No pretrained models found. Please run one of these scripts to download models:
        ```
        python download_model.py            # For HuggingFace ViT model
        python download_advanced_model.py   # For advanced EfficientNet-B4
        ```
        Using fallback model for now (may have reduced accuracy).
        """)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # File type selection
    file_type = st.sidebar.radio("Select File Type", ["Image", "Video"])
    
    # Model selection options
    model_options = ["huggingface_vit", "efficientnet_b3", "efficientnet_b0"]
    
    # Default to HuggingFace ViT model if available
    default_model = 0 if os.path.exists(HUGGINGFACE_MODEL_DIR) else 1
        
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model", 
        model_options,
        index=default_model,
        help="Select the model architecture for deepfake detection"
    )
    
    # Video settings
    if file_type == "Video":
        max_frames = st.sidebar.slider(
            "Number of Frames to Process", 
            min_value=10, 
            max_value=300, 
            value=30,
            help="Higher values will process more frames but take longer"
        )
    
    # Load model
    with st.spinner("Loading model..."):
        face_detector, preprocessor, model, device, is_advanced_model = load_detection_model(model_name)
    
    # File uploader
    if file_type == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Process the image based on model type
            with st.spinner("Processing image..."):
                if is_advanced_model:
                    # Process with advanced model
                    result_image, detection_results = process_image_advanced(
                        uploaded_file, model
                    )
                else:
                    # Process with standard model
                    result_image, detection_results = process_image(
                        uploaded_file, face_detector, preprocessor, model
                    )
            
            # Display original and processed images side by side
            col1, col2 = st.columns(2)
            col1.header("Original Image")
            col1.image(uploaded_file, use_container_width=True)
            
            col2.header("Detection Result")
            col2.image(result_image, use_container_width=True)
            
            # Show prediction results
            st.header("Detection Results")
            show_prediction_results(detection_results)
            
            # Display clear verdict
            display_verdict(detection_results["fake_probabilities"])
    
    else:  # Video
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_file is not None:
            # Save the uploaded video to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name
            
            if is_advanced_model:
                # Process with advanced model
                with st.spinner(f"Processing video with advanced model ({max_frames} frames)... This may take some time."):
                    start_time = time.time()
                    
                    # Process the video with advanced model
                    results = model.detect_from_video(
                        video_path,
                        num_frames=max_frames,
                        save_visualization=True
                    )
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                
                # Display processed video if visualization was saved
                if "visualization_path" in results and os.path.exists(results["visualization_path"]):
                    st.header("Processed Video")
                    st.video(results["visualization_path"])
                else:
                    st.header("Original Video")
                    st.video(video_path)
                
                # Display processing stats
                st.header("Processing Information")
                st.write(f"Processing time: {processing_time:.2f} seconds")
                st.write(f"Analyzed {results['analyzed_frames']} frames")
                
                # Show video stats
                st.header("Detection Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Fake Probability", f"{results['mean_score']:.2f}")
                col2.metric("Max Fake Probability", f"{results['max_score']:.2f}")
                col3.metric("Confidence", f"{results['confidence']:.2f}")
                
                # Display clear verdict
                st.markdown("---")
                st.header("VERDICT")
                
                # Create columns for layout
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    # Display icon
                    if results["is_deepfake"]:
                        st.markdown("### 🔴")
                    else:
                        st.markdown("### 🟢")
                
                with col2:
                    # Display verdict text with appropriate color
                    if results["is_deepfake"]:
                        st.markdown(f'<h2 style="color: red;">FAKE DETECTED</h2>', unsafe_allow_html=True)
                        st.markdown(f'<p>Mean fake probability: <b>{results["mean_score"]:.2f}</b></p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<h2 style="color: green;">LIKELY REAL</h2>', unsafe_allow_html=True)
                        st.markdown(f'<p>Mean fake probability: <b>{results["mean_score"]:.2f}</b></p>', unsafe_allow_html=True)
                    
                    # Display confidence level
                    st.progress(results["confidence"], text=f"Confidence: {results['confidence']:.2%}")
            
            else:
                # Initialize standard video processor
                video_processor = VideoProcessor(face_detector, preprocessor, model, device=device)
                
                # Process video with standard model
                with st.spinner(f"Processing video ({max_frames} frames)... This may take some time."):
                    start_time = time.time()
                    
                    # Process the video
                    output_path, video_stats = video_processor.process_video(
                        video_path, 
                        max_frames=max_frames,
                        show_progress=False
                    )
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # Extract some frames for display
                    sample_frames = video_processor.extract_video_frames(video_path, max_frames=9)
                
                # Display processed video
                st.header("Processed Video")
                st.video(output_path)
                
                # Display processing stats
                st.header("Processing Information")
                st.write(f"Processing time: {processing_time:.2f} seconds")
                st.write(f"Processed {video_stats['processed_frames']} frames out of {video_stats['total_frames']} total frames")
                
                # Show video stats
                st.header("Detection Results")
                video_detection_results = {
                    "faces_detected": video_stats['total_faces_detected'],
                    "fake_probabilities": video_stats['fake_probabilities']
                }
                show_prediction_results(video_detection_results)
                
                # Display clear verdict for video
                display_verdict(video_stats['fake_probabilities'])
                
                # Display additional video stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Frames with Faces", video_stats['frames_with_faces'])
                col2.metric("Max Fake Probability", f"{video_stats['max_fake_probability']:.2f}")
                col3.metric("Avg Fake Probability", f"{video_stats['avg_fake_probability']:.2f}")
                
                # Display sample frames
                st.header("Sample Frames")
                display_video_frames(sample_frames)
            
            # Clean up temporary file
            try:
                os.unlink(video_path)
            except PermissionError:
                # File is still being used by Streamlit for playback
                st.info("Note: Temporary video file will be cleaned up later.")
                pass
            except Exception as e:
                st.warning(f"Could not remove temporary file: {e}")
                pass
    
    # Show app info
    st.sidebar.markdown("---")
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This application uses deep learning to detect deepfake content in images and videos.
        
        **Pretrained Models:**
        - Advanced EfficientNet-B4 with specialized deepfake weights
        - Vision Transformer (ViT) from HuggingFace
        - EfficientNet (B0/B3) with ImageNet weights
        - MTCNN for face detection
        
        **Technologies:**
        - PyTorch
        - OpenCV
        - Streamlit
        
        **Specialized Datasets:**
        - Advanced models are trained on FaceForensics++, CelebDF, DFDC
        """
    )

if __name__ == "__main__":
    main() 