import cv2
import time
import sys

# Import components, handling potential package execution paths
try:
    from src.camera import Camera
    from src.detection import PersonDetector
except ModuleNotFoundError:
    from camera import Camera
    from detection import PersonDetector

def main():
    # Initialize the person detector
    print("Initializing YOLOv8n detector...")
    # By default, YOLOv8n (yolov8n.pt) weights will download automatically if not found locally
    detector = PersonDetector(model_path="yolov8n.pt", confidence_threshold=0.5)
    
    # Initialize the camera wrapper with default device index (0)
    camera = Camera(device_index=0)
    
    print("Attempting to open default webcam...")
    if not camera.open():
        print("Error: Could not open the webcam.")
        print("Please check that your webcam is plugged in, active, and not in use by another application.")
        return

    print("Webcam opened successfully. Real-time person detection is active.")
    print("Press 'q' inside the video window to quit.")
    
    # Initialize timer for FPS calculation
    prev_time = time.time()
    
    try:
        while True:
            # Read a frame from the camera
            ret, frame = camera.read_frame()
            if not ret:
                print("Error: Failed to capture frame from webcam. Exiting...")
                break
            
            # Detect people
            detections = detector.detect(frame)
            
            # Draw bounding boxes and confidence labels on the frame
            frame = detector.draw_detections(frame, detections)
            
            # Calculate instantaneous FPS
            curr_time = time.time()
            elapsed_time = curr_time - prev_time
            fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
            prev_time = curr_time
            
            # Draw premium semi-transparent HUD overlay
            # 1. Overlay a dark HUD background panel
            overlay = frame.copy()
            hud_x1, hud_y1 = 15, 15
            hud_x2, hud_y2 = 250, 115
            cv2.rectangle(overlay, (hud_x1, hud_y1), (hud_x2, hud_y2), (0, 0, 0), cv2.FILLED)
            
            # Draw overlay with 60% transparency
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # 2. Add HUD Text Details
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Draw HUD Title
            cv2.putText(
                frame, 
                "NAVIGO MONITOR", 
                (hud_x1 + 10, hud_y1 + 22), 
                font, 
                0.55, 
                (255, 191, 0),  # Electric cyan-blue
                2, 
                cv2.LINE_AA
            )
            
            # Draw separation line
            cv2.line(
                frame, 
                (hud_x1 + 10, hud_y1 + 30), 
                (hud_x2 - 10, hud_y1 + 30), 
                (100, 100, 100), 
                1, 
                cv2.LINE_AA
            )
            
            # Draw FPS
            cv2.putText(
                frame, 
                f"SYS FPS: {fps:.1f}", 
                (hud_x1 + 10, hud_y1 + 52), 
                font, 
                0.45, 
                (255, 255, 255), 
                1, 
                cv2.LINE_AA
            )
            
            # Draw Person Count
            people_count = len(detections)
            cv2.putText(
                frame, 
                f"TARGETS: {people_count}", 
                (hud_x1 + 10, hud_y1 + 75), 
                font, 
                0.45, 
                (255, 255, 255), 
                1, 
                cv2.LINE_AA
            )
            
            # Draw status / mode
            cv2.putText(
                frame, 
                "MODE: Person Tracking", 
                (hud_x1 + 10, hud_y1 + 95), 
                font, 
                0.35, 
                (0, 255, 255),  # Yellow status indicator
                1, 
                cv2.LINE_AA
            )
            
            # Display the resulting frame in the window
            cv2.imshow("NAVIGO - Webcam Feed (Phase 2)", frame)
            
            # Listen for keyboard input. Wait 1ms per frame, check if key is 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed. Initiating shutdown...")
                break
                
    except KeyboardInterrupt:
        print("\nExecution interrupted by user keyboard interrupt.")
        
    finally:
        # Clean up all resources
        print("Cleaning up resources...")
        camera.release()
        cv2.destroyAllWindows()
        print("NAVIGO shutdown complete.")

if __name__ == "__main__":
    main()
