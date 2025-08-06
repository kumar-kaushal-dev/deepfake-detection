import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import cv2

class SpecializedDeepfakeDetector:
    """
    A specialized deepfake detector using the prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0 model
    which can differentiate between AI-generated images, deepfakes, and real images.
    """
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        print("Loading specialized deepfake detection model...")
        self.processor = AutoImageProcessor.from_pretrained("prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0")
        self.model = AutoModelForImageClassification.from_pretrained("prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get the label mappings
        self.id2label = self.model.config.id2label
        print(f"Model loaded successfully with labels: {self.id2label}")
    
    def detect_from_frame(self, frame):
        """
        Detect if a frame is AI-generated, deepfake, or real
        
        Args:
            frame: The input image/frame (numpy array or PIL Image)
            
        Returns:
            dict: Detailed prediction results including probabilities for each class
        """
        # Convert to PIL Image if numpy array
        if isinstance(frame, np.ndarray):
            if frame.shape[2] == 3:  # Check if BGR (OpenCV format)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            image = frame
        
        # Process the image and get predictions
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        
        # Convert to prediction dictionary
        predictions = {}
        for i, prob in enumerate(probabilities):
            label = self.id2label[i]
            predictions[label] = prob.item()
        
        # Also determine the most likely class
        most_likely = self.id2label[probabilities.argmax().item()]
        predictions["most_likely"] = most_likely
        
        # For compatibility with other detectors, add a 'fake_probability'
        # Consider both AI and Deepfake as fake
        fake_prob = predictions.get("AI", 0) + predictions.get("Deepfake", 0)
        predictions["fake_probability"] = fake_prob
        
        return predictions
    
    def detect_from_video(self, video_path, num_frames=30, save_visualization=False, output_path=None):
        """
        Detect deepfakes in a video by analyzing multiple frames
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to analyze
            save_visualization: Whether to save a visualization of the results
            output_path: Path to save the visualization video
            
        Returns:
            Dictionary with prediction results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to analyze
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            # Evenly distribute frame selection
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        predictions = []
        analyzed_frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                pred = self.detect_from_frame(frame)
                predictions.append(pred)
                
                if save_visualization:
                    # Add prediction text to frame
                    most_likely = pred["most_likely"]
                    real_prob = pred.get("Real", 0)
                    ai_prob = pred.get("AI", 0)
                    deepfake_prob = pred.get("Deepfake", 0)
                    
                    # Choose color based on prediction (green for real, red for fake)
                    if most_likely == "Real":
                        color = (0, 255, 0)  # Green for real
                    else:
                        color = (0, 0, 255)  # Red for fake
                    
                    # Add text to the frame
                    cv2.putText(frame, f"Class: {most_likely}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"Real: {real_prob:.2f}", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"AI: {ai_prob:.2f}", (10, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Deepfake: {deepfake_prob:.2f}", (10, 130), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    analyzed_frames.append(frame)
            else:
                print(f"Could not read frame {idx}")
        
        cap.release()
        
        # Calculate statistics
        fake_probs = [p["fake_probability"] for p in predictions]
        mean_fake = sum(fake_probs) / len(fake_probs) if fake_probs else 0
        max_fake = max(fake_probs) if fake_probs else 0
        
        # Get most frequent classification
        class_counts = {"Real": 0, "AI": 0, "Deepfake": 0}
        for p in predictions:
            class_counts[p["most_likely"]] = class_counts.get(p["most_likely"], 0) + 1
        
        most_common_class = max(class_counts.items(), key=lambda x: x[1])[0]
        
        # Is it a deepfake based on the most common class?
        is_deepfake = most_common_class != "Real"
        
        # Prepare results
        result = {
            "is_deepfake": is_deepfake,
            "confidence": mean_fake if is_deepfake else (1 - mean_fake),
            "mean_fake_probability": mean_fake,
            "max_fake_probability": max_fake,
            "most_likely_class": most_common_class,
            "class_distribution": class_counts,
            "analyzed_frames": len(predictions)
        }
        
        # Save visualization video if requested
        if save_visualization and analyzed_frames:
            if output_path is None:
                base, ext = os.path.splitext(video_path)
                output_path = f"{base}_analyzed{ext}"
            
            height, width = analyzed_frames[0].shape[:2]
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps/2, (width, height))
            
            for frame in analyzed_frames:
                out.write(frame)
            
            out.release()
            result["visualization_path"] = output_path
        
        return result

def download_model():
    """
    Download the specialized deepfake detection model
    
    Returns:
        str: Path to the model directory
    """
    # Create directory if it doesn't exist
    model_dir = os.path.join('pretrained_models', 'specialized_deepfake_model')
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Downloading specialized deepfake detection model to {model_dir}...")
    
    # This will download and cache the model locally
    # Use AutoImageProcessor instead of AutoProcessor
    processor = AutoImageProcessor.from_pretrained("prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0", cache_dir=model_dir)
    model = AutoModelForImageClassification.from_pretrained("prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0", cache_dir=model_dir)
    
    print(f"Model downloaded successfully with labels: {model.config.id2label}")
    print(f"Model saved to {model_dir}")
    
    return model_dir

# Test the model on a sample image
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the specialized deepfake detection model")
    parser.add_argument("--download", action="store_true", help="Download the model")
    parser.add_argument("--image", type=str, help="Path to an image to test")
    
    args = parser.parse_args()
    
    if args.download:
        download_model()
    
    # Create a detector
    detector = SpecializedDeepfakeDetector()
    
    # If an image path is provided, test the detector on it
    if args.image and os.path.exists(args.image):
        image = Image.open(args.image).convert("RGB")
        predictions = detector.detect_from_frame(image)
        
        print("\nPredictions:")
        for label, prob in predictions.items():
            if label != "most_likely":
                print(f"  {label}: {prob:.4f}")
        
        print(f"\nMost likely class: {predictions['most_likely']}")
        print(f"Fake probability: {predictions['fake_probability']:.4f}")
    else:
        # Test on random images
        print("Testing on random images...")
        
        # Create a random noise image
        noise_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        noise_pred = detector.detect_from_frame(noise_image)
        
        # Create a gray image
        gray_image = np.ones((224, 224, 3), dtype=np.uint8) * 128
        gray_pred = detector.detect_from_frame(gray_image)
        
        print("\nRandom noise image predictions:")
        for label, prob in noise_pred.items():
            if label != "most_likely":
                print(f"  {label}: {prob:.4f}")
        
        print(f"\nGray image predictions:")
        for label, prob in gray_pred.items():
            if label != "most_likely":
                print(f"  {label}: {prob:.4f}")
        
        print("\nModel is ready for use with the app!")
        print("Run: streamlit run app.py")
        print("And select the 'huggingface_vit' model in the app.") 