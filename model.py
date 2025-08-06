import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import os
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np

class DeepfakeClassifier(nn.Module):
    """
    EfficientNet-based model for deepfake detection
    """
    def __init__(self, model_name='efficientnet_b0', num_classes=1):
        super(DeepfakeClassifier, self).__init__()
        
        # Load pre-trained EfficientNet
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights="DEFAULT")
        elif model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(weights="DEFAULT")
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Get the number of features in the last layer
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier with a custom one
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=num_features, out_features=num_classes)
        )
        
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        features = self.backbone(x)
        # Apply sigmoid activation for binary classification (0-1 range)
        return self.sigmoid(features)

class HuggingFaceDeepfakeDetector(nn.Module):
    """
    Adapter for HuggingFace Vision Transformer deepfake detection model
    """
    def __init__(self, model_path):
        super(HuggingFaceDeepfakeDetector, self).__init__()
        
        # Load model and processor
        self.model = ViTForImageClassification.from_pretrained(model_path)
        self.processor = ViTImageProcessor.from_pretrained(model_path)
        
        # Labels mapping (typically 0 = real, 1 = fake)
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        # Find the fake label index
        self.fake_idx = None
        for idx, label in self.id2label.items():
            if label.lower() in ['fake', 'deepfake', 'synthetic', 'ai', 'false']:
                self.fake_idx = int(idx)
                break
        
        # Default to class 1 if no clear fake label found
        if self.fake_idx is None:
            self.fake_idx = 1
            
    def preprocess(self, images):
        """
        Preprocess images for the ViT model
        
        Args:
            images: List of images or batch tensor
            
        Returns:
            dict: Preprocessed inputs for the model
        """
        # If we get a list of numpy arrays, convert to tensor first
        if isinstance(images, list) and isinstance(images[0], np.ndarray):
            return self.processor(images=images, return_tensors="pt")
        
        # If we already have a tensor batch, we need to denormalize and process
        if torch.is_tensor(images):
            # The processor expects PIL images or numpy arrays
            # We can't easily convert back, so we'll create a custom preprocessing
            # for already preprocessed tensors
            return {"pixel_values": images}
        
        return self.processor(images=images, return_tensors="pt")
        
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Preprocessed input tensor or batch of tensors
            
        Returns:
            torch.Tensor: Probability that the image is fake (0-1)
        """
        # If x is already a dict with pixel_values, use it directly
        if isinstance(x, dict) and "pixel_values" in x:
            inputs = x
        else:
            # Otherwise, preprocess the input
            inputs = self.preprocess(x)
            
        # Move inputs to the same device as the model
        for key, val in inputs.items():
            if torch.is_tensor(val):
                inputs[key] = val.to(self.model.device)
                
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Return probability of the fake class (typically index 1)
        return probs[:, self.fake_idx]

class Preprocessor:
    """
    Image preprocessor for the deepfake detection model
    """
    def __init__(self, image_size=(224, 224)):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess(self, image):
        """
        Preprocess a single image
        
        Args:
            image (np.ndarray): Input image in RGB format
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        return self.transform(image)
    
    def preprocess_batch(self, images):
        """
        Preprocess a batch of images
        
        Args:
            images (list): List of images in RGB format
            
        Returns:
            torch.Tensor: Batch of preprocessed image tensors
        """
        batch = torch.stack([self.preprocess(img) for img in images])
        return batch

def load_model(model_path=None, model_name='efficientnet_b0', device='cpu'):
    """
    Load the deepfake detection model
    
    Args:
        model_path (str): Path to the saved model
        model_name (str): Name of the model architecture
        device (str): Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        DeepfakeClassifier: Loaded model
    """
    # Check if model_path is a HuggingFace ViT model directory
    if model_path and os.path.isdir(model_path) and (
        os.path.exists(os.path.join(model_path, "config.json")) or
        os.path.exists(os.path.join(model_path, "pytorch_model.bin")) or
        os.path.exists(os.path.join(model_path, "model.safetensors"))
    ):
        try:
            # Load HuggingFace model
            model = HuggingFaceDeepfakeDetector(model_path)
            model = model.to(device)
            model.eval()
            print(f"Loaded HuggingFace ViT model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading HuggingFace model: {e}")
            print("Falling back to EfficientNet model...")
    
    # Load traditional model (EfficientNet)
    model = DeepfakeClassifier(model_name=model_name)
    
    # Load pre-trained weights if provided
    if model_path:
        try:
            # First check if file exists
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
            else:
                # Try to read the first few bytes to check if it's a valid model file
                with open(model_path, 'rb') as f:
                    header = f.read(10)
                    # Check if it looks like an HTML file
                    if header.startswith(b'<'):
                        print(f"Invalid model file (looks like HTML): {model_path}")
                    else:
                        # Attempt to load the model
                        state_dict = torch.load(model_path, map_location=device)
                        model.load_state_dict(state_dict)
                        print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            print("Using model with default ImageNet weights instead")
    else:
        print("No model path provided. Using model with default ImageNet weights.")
    
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    return model 