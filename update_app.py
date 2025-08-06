import streamlit as st
import os
import tempfile
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import time

# Try to import the specialized model
try:
    from specialized_model import SpecializedDeepfakeDetector
    SPECIALIZED_MODEL_AVAILABLE = True
except ImportError:
    SPECIALIZED_MODEL_AVAILABLE = False
    st.error("Specialized model module not available. Please run `python specialized_model.py --download`")

# Page configuration
st.set_page_config(
    page_title="Advanced Deepfake Detection",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
TEMP_DIR = tempfile.gettempdir()
UPLOAD_DIR = "uploaded_files"

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Function to load the specialized model
@st.cache_resource
def load_specialized_model():
    """Load the specialized deepfake detection model"""
    if SPECIALIZED_MODEL_AVAILABLE:
        return SpecializedDeepfakeDetector()
    else:
        return None

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
def display_verdict(prediction_results):
    """
    Display a clear verdict about whether content is fake or real
    
    Args:
        prediction_results (dict): Prediction results
    """
    # Extract results
    most_likely_class = prediction_results.get("most_likely_class", "Unknown")
    is_fake = most_likely_class != "Real"
    fake_probability = prediction_results.get("fake_probability", 0.5)
    confidence = prediction_results.get("confidence", 0.5)
    class_distribution = prediction_results.get("class_distribution", {})
    
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
            st.markdown(f'<h2 style="color: red;">{most_likely_class.upper()} DETECTED</h2>', unsafe_allow_html=True)
            st.markdown(f'<p>Fake probability: <b>{fake_probability:.2f}</b></p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<h2 style="color: green;">LIKELY REAL</h2>', unsafe_allow_html=True)
            st.markdown(f'<p>Real confidence: <b>{1-fake_probability:.2f}</b></p>', unsafe_allow_html=True)
        
        # Display confidence level
        st.progress(confidence, text=f"Confidence: {confidence:.2%}")
    
    # Display class distribution
    st.subheader("Classification Distribution")
    class_data = {"Class": list(class_distribution.keys()), "Count": list(class_distribution.values())}
    
    # Create a horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(class_data["Class"], class_data["Count"], color=['green' if c == 'Real' else 'red' for c in class_data["Class"]])
    ax.set_xlabel('Count')
    ax.set_title('Classification Distribution')
    for i, v in enumerate(class_data["Count"]):
        ax.text(v + 0.1, i, str(v), color='black', va='center')
    st.pyplot(fig)

# Function to process an uploaded image
def process_image(image_file, detector):
    """
    Process an uploaded image for deepfake detection
    
    Args:
        image_file: Uploaded image file
        detector: Deepfake detector
        
    Returns:
        tuple: (result_image, detection_results)
    """
    # Read the image
    image = Image.open(image_file).convert('RGB')
    image_np = np.array(image)
    
    # Process the image
    predictions = detector.detect_from_frame(image_np)
    
    # Get the most likely class and probabilities
    most_likely = predictions["most_likely"]
    real_prob = predictions.get("Real", 0)
    ai_prob = predictions.get("AI", 0)
    deepfake_prob = predictions.get("Deepfake", 0)
    fake_prob = predictions["fake_probability"]
    
    # Create result image with prediction
    result_image = image_np.copy()
    
    # Choose color based on prediction (green for real, red for fake)
    if most_likely == "Real":
        color = (0, 255, 0)  # BGR format (OpenCV)
    else:
        color = (0, 0, 255)  # BGR format (OpenCV)
    
    # Add text showing prediction
    cv2.putText(result_image, f"Class: {most_likely}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(result_image, f"Real: {real_prob:.2f}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(result_image, f"AI: {ai_prob:.2f}", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(result_image, f"Deepfake: {deepfake_prob:.2f}", (10, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Convert BGR back to RGB for display
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    # Prepare detection results
    detection_results = {
        "most_likely_class": most_likely,
        "fake_probability": fake_prob,
        "confidence": max(fake_prob, 1-fake_prob),
        "class_distribution": {
            "Real": real_prob,
            "AI": ai_prob,
            "Deepfake": deepfake_prob
        }
    }
    
    return result_rgb, detection_results

def main():
    # Add title and description
    st.title("🕵️‍♀️ Advanced Deepfake Detection")
    st.markdown("""
    This application can detect deepfake and AI-generated content in images and videos using a specialized model.
    The model can differentiate between real images, AI-generated images, and deepfakes.
    Upload an image or video file to get started.
    """)
    
    # Check if the specialized model is available
    if not SPECIALIZED_MODEL_AVAILABLE:
        st.error("""
        Specialized model not available. Please run the following command to set it up:
        ```
        python specialized_model.py --download
        ```
        """)
        st.stop()
    
    # Load the model
    with st.spinner("Loading specialized deepfake detection model..."):
        detector = load_specialized_model()
        
        if detector is None:
            st.error("Failed to load the specialized model.")
            st.stop()
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # File type selection
    file_type = st.sidebar.radio("Select File Type", ["Image", "Video"])
    
    # Video settings
    if file_type == "Video":
        max_frames = st.sidebar.slider(
            "Number of Frames to Process", 
            min_value=10, 
            max_value=100, 
            value=30,
            help="Higher values will process more frames but take longer"
        )
    
    # File uploader
    if file_type == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Process the image
            with st.spinner("Processing image..."):
                result_image, detection_results = process_image(uploaded_file, detector)
            
            # Display original and processed images side by side
            col1, col2 = st.columns(2)
            col1.header("Original Image")
            col1.image(uploaded_file, use_container_width=True)
            
            col2.header("Detection Result")
            col2.image(result_image, use_container_width=True)
            
            # Display predictions
            st.header("Detailed Results")
            
            # Extract probabilities 
            real_prob = detection_results["class_distribution"]["Real"]
            ai_prob = detection_results["class_distribution"]["AI"]
            deepfake_prob = detection_results["class_distribution"]["Deepfake"]
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Real Probability", f"{real_prob:.2f}")
            col2.metric("AI Generated Probability", f"{ai_prob:.2f}")
            col3.metric("Deepfake Probability", f"{deepfake_prob:.2f}")
            
            # Plot probabilities
            fig, ax = plt.subplots(figsize=(10, 4))
            classes = ["Real", "AI", "Deepfake"]
            probs = [real_prob, ai_prob, deepfake_prob]
            colors = ['green', 'red', 'orange']
            
            ax.bar(classes, probs, color=colors)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            ax.set_title('Classification Probabilities')
            for i, v in enumerate(probs):
                ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
            
            st.pyplot(fig)
            
            # Display clear verdict
            display_verdict(detection_results)
    
    else:  # Video
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_file is not None:
            # Save the uploaded video to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name
            
            # Process the video
            with st.spinner(f"Processing video ({max_frames} frames)... This may take some time."):
                start_time = time.time()
                
                # Process the video
                results = detector.detect_from_video(
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
            
            # Get distribution
            class_distribution = results["class_distribution"]
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Real Frames", class_distribution.get("Real", 0))
            col2.metric("AI Generated Frames", class_distribution.get("AI", 0))
            col3.metric("Deepfake Frames", class_distribution.get("Deepfake", 0))
            
            # Display more metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Most Likely Class", results["most_likely_class"])
            col2.metric("Mean Fake Probability", f"{results['mean_fake_probability']:.2f}")
            col3.metric("Max Fake Probability", f"{results['max_fake_probability']:.2f}")
            
            # Display clear verdict
            display_verdict(results)
            
            # Clean up temporary file
            try:
                os.unlink(video_path)
                if "visualization_path" in results:
                    os.unlink(results["visualization_path"])
            except:
                pass
    
    # Show app info
    st.sidebar.markdown("---")
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This application uses a specialized model to detect deepfake and AI-generated content.
        
        **Model:**
        - prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0
        
        **Features:**
        - Can distinguish between real media, AI-generated content, and deepfakes
        - Works on both images and videos
        - Provides detailed analysis with confidence scores
        
        **Technology:**
        - Hugging Face Transformers
        - PyTorch
        - OpenCV
        - Streamlit
        """
    )

if __name__ == "__main__":
    main() 