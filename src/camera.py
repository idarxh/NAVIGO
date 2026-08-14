import cv2

class Camera:
    """
    A reusable wrapper around OpenCV's VideoCapture to handle webcam operations safely.
    """
    def __init__(self, device_index: int = 0):
        """
        Initialize the Camera instance.
        
        Args:
            device_index (int): The system index of the video capture device. Default is 0.
        """
        self.device_index = device_index
        self.cap = None

    def open(self) -> bool:
        """
        Attempts to open the camera device.
        
        Returns:
            bool: True if the camera opened successfully, False otherwise.
        """
        self.cap = cv2.VideoCapture(self.device_index)
        return self.is_opened()

    def is_opened(self) -> bool:
        """
        Checks if the camera device is currently open and ready.
        
        Returns:
            bool: True if open, False otherwise.
        """
        if self.cap is None:
            return False
        return self.cap.isOpened()

    def read_frame(self):
        """
        Reads a single frame from the camera.
        
        Returns:
            tuple: (ret, frame) where ret is a boolean indicating success, and frame is the image array.
        """
        if not self.is_opened():
            return False, None
        return self.cap.read()

    def release(self):
        """
        Safely releases the camera hardware and resets the VideoCapture instance.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
