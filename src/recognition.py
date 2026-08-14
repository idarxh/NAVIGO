import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class OwnerRecognizer:
    """
    A class to perform owner recognition on cropped person images using a MobileNetV3-based classifier.
    """
    def __init__(self, model_path: str = "models/owner_classifier.pth", device: str = None):
        """
        Initialize the OwnerRecognizer.
        
        Args:
            model_path (str): Path to the trained PyTorch model weights.
            device (str): Device to run inference on ('cpu', 'cuda', etc.).
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.classes = ["OTHER", "OWNER"]  # 0: OTHER/others, 1: OWNER/owner
        
        # Instantiate MobileNetV3 Small architecture
        self.model = models.mobilenet_v3_small(weights=None)
        
        # Modify the classification head to output binary classes (OTHER vs OWNER)
        num_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_features, len(self.classes))
        
        self.model.to(self.device)
        self.loaded = False
        
        # Load weights if the file exists
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f"OwnerRecognizer: Loaded trained model weights from '{self.model_path}'.")
                self.loaded = True
            except Exception as e:
                print(f"Error: Failed to load model weights from '{self.model_path}': {e}")
                print("OwnerRecognizer: Running in dummy/untrained mode.")
        else:
            print(f"Warning: Model weights not found at '{self.model_path}'.")
            print("OwnerRecognizer: Running in dummy/untrained mode.")
            
        self.model.eval()
        
        # Define validation transforms (matching those used during training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, crop_frame):
        """
        Predict whether a cropped person image is OWNER or OTHER.
        
        Args:
            crop_frame (numpy.ndarray): Cropped BGR image array from OpenCV.
            
        Returns:
            tuple: (label (str), confidence (float))
        """
        if crop_frame is None or crop_frame.size == 0:
            return "OTHER", 0.0
            
        try:
            # Convert BGR frame (OpenCV) to RGB (PIL)
            rgb_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Apply preprocessing transforms
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                confidence, class_idx = torch.max(probabilities, dim=0)
                
            label = self.classes[class_idx.item()]
            return label, confidence.item()
            
        except Exception as e:
            print(f"Error during owner recognition inference: {e}")
            return "OTHER", 0.0
