import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch

class FaceDetector:
    def __init__(self, device='cpu'):
        """
        Initialize face detector using MTCNN from facenet-pytorch
        
        Args:
            device (str): Device to use for detection ('cpu' or 'cuda')
        """
        self.device = device
        self.mtcnn = MTCNN(
            margin=40, 
            select_largest=False, 
            post_process=False,
            device=self.device
        )
        
    def detect_faces(self, image):
        """
        Detect faces in an image
        
        Args:
            image (np.ndarray): Input image in BGR format (OpenCV default)
            
        Returns:
            tuple: (face_crops, bboxes) where face_crops is a list of cropped faces
                  and bboxes is a list of bounding boxes in format [x1, y1, x2, y2]
        """
        # Convert BGR to RGB (MTCNN expects RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        # Detect faces
        boxes, _ = self.mtcnn.detect(rgb_image)
        
        # Handle case where no faces are detected
        if boxes is None:
            return [], []
        
        face_crops = []
        valid_boxes = []
        
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Check if the box is valid
            if x1 < 0 or y1 < 0 or x2 >= image.shape[1] or y2 >= image.shape[0]:
                continue
                
            # Extract face
            face = rgb_image[y1:y2, x1:x2]
            
            # Only keep valid faces
            if face.size > 0 and face.shape[0] > 0 and face.shape[1] > 0:
                face_crops.append(face)
                valid_boxes.append([x1, y1, x2, y2])
                
        return face_crops, valid_boxes
    
    def extract_faces_from_frame(self, frame, size=(224, 224)):
        """
        Extract and preprocess faces from a video frame
        
        Args:
            frame (np.ndarray): Video frame
            size (tuple): Output size for the face crops
            
        Returns:
            tuple: (faces, boxes) where faces is a list of preprocessed face tensors
                  and boxes is a list of bounding boxes
        """
        # Detect faces
        face_crops, boxes = self.detect_faces(frame)
        
        # Preprocess faces
        processed_faces = []
        for face in face_crops:
            # Resize face crop
            face_resized = cv2.resize(face, size)
            processed_faces.append(face_resized)
            
        return processed_faces, boxes

def draw_detection_results(frame, boxes, predictions):
    """
    Draw detection results on a frame
    
    Args:
        frame (np.ndarray): The video frame to draw on
        boxes (list): List of bounding boxes [x1, y1, x2, y2]
        predictions (list): List of deepfake probabilities for each face
        
    Returns:
        np.ndarray: Frame with bounding boxes and prediction results
    """
    result_frame = frame.copy()
    
    for box, pred in zip(boxes, predictions):
        x1, y1, x2, y2 = box
        
        # Determine color based on prediction (red for fake, green for real)
        # Higher value = more likely to be fake
        color = (0, int(255 * (1 - pred)), int(255 * pred))  # BGR format
        
        # Draw bounding box
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        label = f"Fake: {pred:.2f}"
        
        # Draw label background
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result_frame, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), 
                     color, 
                     cv2.FILLED)
        
        # Draw label text
        cv2.putText(result_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return result_frame 