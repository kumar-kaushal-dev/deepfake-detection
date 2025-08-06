import cv2
import numpy as np
import torch
import tempfile
import os
from tqdm import tqdm
from face_utils import FaceDetector, draw_detection_results
from model import Preprocessor

class VideoProcessor:
    """
    Process videos for deepfake detection
    """
    def __init__(self, face_detector, preprocessor, model, device='cpu'):
        """
        Initialize the video processor
        
        Args:
            face_detector (FaceDetector): Face detector
            preprocessor (Preprocessor): Image preprocessor
            model: Deepfake detection model
            device (str): Device to run inference on ('cpu' or 'cuda')
        """
        self.face_detector = face_detector
        self.preprocessor = preprocessor
        self.model = model
        self.device = device
        
    def process_frame(self, frame):
        """
        Process a single frame for deepfake detection
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: (result_frame, frame_stats) where result_frame is the frame with
                  detection results drawn on it and frame_stats contains statistics
        """
        # Extract faces from the frame
        faces, boxes = self.face_detector.extract_faces_from_frame(frame)
        
        # If no faces found, return the original frame and empty stats
        if not faces:
            return frame, {'faces_detected': 0, 'fake_probabilities': []}
        
        # Preprocess the faces
        face_tensors = self.preprocessor.preprocess_batch(faces)
        face_tensors = face_tensors.to(self.device)
        
        # Get predictions from the model
        with torch.no_grad():
            predictions = self.model(face_tensors).cpu().numpy().flatten()
        
        # Draw detection results on the frame
        result_frame = draw_detection_results(frame, boxes, predictions)
        
        # Collect statistics
        frame_stats = {
            'faces_detected': len(faces),
            'fake_probabilities': predictions.tolist()
        }
        
        return result_frame, frame_stats
    
    def process_video(self, video_path, output_path=None, max_frames=30, frame_interval=None, show_progress=True):
        """
        Process a video for deepfake detection
        
        Args:
            video_path (str): Path to the input video
            output_path (str): Path to save the output video. If None, a temporary file is created.
            max_frames (int): Maximum number of frames to process
            frame_interval (int): Interval between frames to process. If None, calculated based on max_frames.
            show_progress (bool): Whether to show a progress bar
            
        Returns:
            tuple: (output_path, video_stats) where output_path is the path to the processed video
                  and video_stats contains statistics about the detections
        """
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval if not provided
        if frame_interval is None:
            frame_interval = max(1, total_frames // max_frames)
        
        frames_to_process = min(max_frames, total_frames)
        
        # Create a temporary output file if not provided
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, 'deepfake_detection_result.mp4')
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize statistics
        video_stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'total_faces_detected': 0,
            'fake_probabilities': [],
            'frames_with_faces': 0,
            'max_fake_probability': 0.0,
            'avg_fake_probability': 0.0
        }
        
        # Process video frames
        progress_bar = None
        if show_progress:
            progress_bar = tqdm(total=frames_to_process, desc="Processing video")
        
        frame_idx = 0
        while cap.isOpened() and video_stats['processed_frames'] < frames_to_process:
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Process frame at intervals
            if frame_idx % frame_interval == 0:
                # Process the frame
                result_frame, frame_stats = self.process_frame(frame)
                
                # Update statistics
                video_stats['processed_frames'] += 1
                video_stats['total_faces_detected'] += frame_stats['faces_detected']
                
                if frame_stats['faces_detected'] > 0:
                    video_stats['frames_with_faces'] += 1
                    video_stats['fake_probabilities'].extend(frame_stats['fake_probabilities'])
                    
                    # Update max fake probability
                    max_prob = max(frame_stats['fake_probabilities'])
                    video_stats['max_fake_probability'] = max(video_stats['max_fake_probability'], max_prob)
                
                # Write the frame
                out.write(result_frame)
                
                # Update progress bar
                if progress_bar:
                    progress_bar.update(1)
            
            frame_idx += 1
            
            # Skip frames according to interval
            if frame_interval > 1 and frame_idx % frame_interval != 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Calculate average fake probability
        if video_stats['fake_probabilities']:
            video_stats['avg_fake_probability'] = sum(video_stats['fake_probabilities']) / len(video_stats['fake_probabilities'])
        
        # Close progress bar
        if progress_bar:
            progress_bar.close()
        
        # Release video resources
        cap.release()
        out.release()
        
        return output_path, video_stats
    
    def extract_video_frames(self, video_path, max_frames=10, with_detection=True):
        """
        Extract frames from a video for display in the Streamlit app
        
        Args:
            video_path (str): Path to the input video
            max_frames (int): Maximum number of frames to extract
            with_detection (bool): Whether to perform deepfake detection on the frames
            
        Returns:
            list: List of extracted frames with detection results if requested
        """
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval
        frame_interval = max(1, total_frames // max_frames)
        
        # Extract frames
        frames = []
        frame_idx = 0
        
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                if with_detection:
                    # Process frame with detection
                    result_frame, _ = self.process_frame(frame)
                    frames.append(result_frame)
                else:
                    # Just add the frame without detection
                    frames.append(frame)
            
            frame_idx += 1
            
            # Skip frames according to interval
            if frame_interval > 1 and frame_idx % frame_interval != 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Release video resources
        cap.release()
        
        return frames 