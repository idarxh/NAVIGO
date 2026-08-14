import cv2
from ultralytics import YOLO

class PersonDetector:
    """
    A class to perform real-time person detection using YOLOv8.
    """
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize the PersonDetector.
        
        Args:
            model_path (str): Path or name of the YOLO model weights. Default is "yolov8n.pt".
            confidence_threshold (float): Minimum confidence threshold for detection. Default is 0.5.
        """
        # Load the YOLO model. This will download the weights automatically if not present locally.
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        # COCO class ID for 'person' is 0
        self.person_class_id = 0

    def detect(self, frame):
        """
        Run inference on the frame and extract person detections.
        
        Args:
            frame: OpenCV image frame (BGR format).
            
        Returns:
            list of dict: List of detections where each detection is:
                          {'box': [x1, y1, x2, y2], 'confidence': score}
        """
        # Run inference (verbose=False keeps the console output clean)
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get the class ID
                cls_id = int(box.cls[0].item())
                # Only keep detections of the 'person' class (ID 0)
                if cls_id == self.person_class_id:
                    conf = float(box.conf[0].item())
                    if conf >= self.confidence_threshold:
                        # Extract xyxy bounding box coordinates
                        xyxy = box.xyxy[0].tolist()
                        detections.append({
                            'box': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                            'confidence': conf
                        })
                        
        return detections

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and confidence labels on the frame.
        
        Args:
            frame: OpenCV image frame (BGR format) to be drawn on.
            detections (list): List of detections to draw.
            
        Returns:
            frame: The annotated image frame.
        """
        for det in detections:
            x1, y1, x2, y2 = det['box']
            conf = det['confidence']
            
            # Premium styling: Vibrant Neon Green (BGR: 57, 255, 20)
            box_color = (20, 255, 57)  # High-contrast spring/neon green
            bg_color = (20, 255, 57)   # Match box color for text banner
            
            # Draw standard bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw label banner
            label = f"Person: {conf:.1%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Get text size to draw background banner
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Adjust label position so it doesn't go off-screen
            label_y1 = max(y1 - text_height - 8, 0)
            label_y2 = y1
            
            # Draw label banner background
            cv2.rectangle(frame, (x1, label_y1), (x1 + text_width + 10, label_y2), bg_color, cv2.FILLED)
            
            # Put text on banner (black text for premium readability on green banner)
            cv2.putText(
                frame, 
                label, 
                (x1 + 5, label_y2 - 4), 
                font, 
                font_scale, 
                (0, 0, 0),  # Black text
                thickness, 
                cv2.LINE_AA
            )
            
        return frame
