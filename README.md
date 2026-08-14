# NAVIGO - Autonomous Smart Luggage Research Project

NAVIGO is an autonomous smart luggage research project designed to track and follow its owner intelligently. The system is engineered to perform computer vision, deep learning classification, and motor control on resource-constrained hardware.

---

## Project Roadmap & Status

### Phase 1: Core Camera Setup (Completed)
- Verifies system integration with the hardware's default webcam.
- Handles safe initialization, frame reading, and clean hardware release.
- Calculates and overlays instantaneous capture frame rate (FPS).

### Phase 2: Real-Time Person Detection (Completed)
- Integrates a lightweight, real-time object detection model (YOLOv8 Nano).
- Filters detections to isolate and track only the `'person'` class (COCO class ID 0).
- Renders target bounding boxes and confidence banners using a high-contrast theme.
- Features a premium, semi-transparent Head-Up Display (HUD) showing tracking telemetry.

### Phase 3: Transfer-Learning Owner Recognition (Planned / Structured)
- **Status**: Code modules and training script implemented; awaiting dataset collection to perform training.
- Uses transfer learning with a pretrained MobileNetV3-Small backbone to classify targets as `OWNER` or `OTHER`.
- Implements stratified validation splitting (80/20 train/val split) and image data augmentations.
- Computes evaluation metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix) and plots training curves.
- Features a reusable `OwnerRecognizer` class that loads saved weights or runs in a safe warning mode if weights are missing.

### Future Research & Core Integrations (Planned)
- **Vision Transformer (ViT) Comparison**: Benchmarking lightweight CNNs against Vision Transformers (ViT) to analyze computational efficiency and accuracy trade-offs on edge devices.
- **Kalman-Filter-Based Tracking**: Integrating coordinate filtering to track the owner through temporary occlusions or crowds.
- **Following Controller & Obstacle Detection**: Designing geometric control algorithms to maintain a steady following distance while navigating obstacles.
- **ESP32 Motor Control**: Implementing serial-based communication between the computer vision system and an ESP32 microcontroller to drive physical luggage wheel motors.

---

## Project Structure

```text
NAVIGO/
├── data/
│   ├── owner/         # Directory for cropped images of the luggage owner.
│   └── others/        # Directory for cropped images of other people.
├── models/            # Stores trained PyTorch model weights (e.g. owner_classifier.pth).
├── experiments/
│   ├── results/       # Stores evaluation metrics, confusion matrices, and training curves.
│   └── train_owner_classifier.py # Pipeline script to train the binary owner recognition model.
├── src/
│   ├── __init__.py    # Marks src as a Python package.
│   ├── camera.py      # Reusable Camera class wrapping OpenCV VideoCapture.
│   ├── detection.py   # Reusable PersonDetector class using YOLOv8.
│   ├── recognition.py # Reusable OwnerRecognizer class using MobileNetV3.
│   └── main.py        # Main execution entrypoint for webcam streaming & HUD.
├── .gitignore         # Configures Git to exclude datasets, weights, caches, and environments.
├── requirements.txt   # Holds project package dependencies.
└── README.md          # Project documentation and quick start guide (this file).
```

---

## Setup and Run Instructions (Windows PowerShell/CMD)

### 1. Set Up Environment & Dependencies
Initialize and activate your environment, then install requirements:
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Live Person Detection (Phase 2)
Start the real-time webcam person detection and telemetry HUD:
```powershell
python src/main.py
```
*Note: Press **'q'** inside the video feed window to stop and close the camera feed.*

### 3. Training the Owner Classifier (Phase 3)
1. Crop images of the owner's body/face and save them under `data/owner/`.
2. Crop images of other people and save them under `data/others/`.
3. Run the training pipeline script:
   ```powershell
   python experiments/train_owner_classifier.py
   ```
   *Note: Training will automatically save the model weights to `models/owner_classifier.pth` and log training evaluation data under `experiments/results/`.*
