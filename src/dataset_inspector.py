import os
import csv
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys
import webbrowser
import threading
import socket
import cv2

# Import the PersonDetector from NAVIGO source files
try:
    from src.detection import PersonDetector
except ModuleNotFoundError:
    from detection import PersonDetector

# Paths
CROPS_DIR = "data/owner"
CSV_PATH = "data/owner_metadata.csv"

# Global detector instance and cache
detector = None
DETECTION_CACHE = {}

def get_detector():
    global detector
    if detector is None:
        print("[Info] Initializing YOLOv8n detector for inspector...")
        detector = PersonDetector(model_path="yolov8n.pt", confidence_threshold=0.5)
    return detector

def get_crops_data():
    if not os.path.exists(CROPS_DIR):
        return []
    
    # Get all jpg/jpeg files in crops directory
    files = [f for f in os.listdir(CROPS_DIR) if f.lower().endswith(('.jpg', '.jpeg'))]
    
    # Load metadata from CSV
    metadata = {}
    if os.path.exists(CSV_PATH):
        try:
            with open(CSV_PATH, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'image_filename' in row:
                        metadata[row['image_filename']] = row
        except Exception as e:
            print(f"[Error] Failed to load metadata CSV: {e}")
            
    crops = []
    for f in sorted(files):
        meta = metadata.get(f, {})
        # Fallback to timestamp if timestamp_seconds doesn't exist
        ts = meta.get('timestamp_seconds') if meta.get('timestamp_seconds') is not None else meta.get('timestamp', 'N/A')
        crops.append({
            'filename': f,
            'source_video': meta.get('source_video', 'Unknown Video'),
            'frame_number': meta.get('frame_number', 'N/A'),
            'timestamp_seconds': ts,
            'yolo_confidence': meta.get('yolo_confidence', 'N/A')
        })
    return crops

def delete_crop(filename):
    filepath = os.path.join(CROPS_DIR, filename)
    file_deleted = False
    
    # Remove from detector cache
    if filename in DETECTION_CACHE:
        del DETECTION_CACHE[filename]
    
    # 1. Delete image file
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            file_deleted = True
        except Exception as e:
            return False, f"System error deleting file: {e}"
    else:
        file_deleted = True  # If it doesn't exist, we treat it as deleted on disk
        
    # 2. Update CSV metadata
    csv_updated = False
    if os.path.exists(CSV_PATH):
        try:
            temp_rows = []
            headers = []
            with open(CSV_PATH, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                for row in reader:
                    if row.get('image_filename') != filename:
                        temp_rows.append(row)
            
            with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(temp_rows)
            csv_updated = True
        except Exception as e:
            return False, f"File deleted, but metadata CSV update failed: {e}"
    else:
        csv_updated = True # No CSV file to update
        
    return (file_deleted and csv_updated), "Crop successfully deleted."

class DatasetInspectorRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence console log messages to keep CLI clean
        return

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        # 1. Serve UI Main Page
        if path in {"/", "/index.html"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            
        # 2. Serve Crops API List
        elif path == "/api/crops":
            crops = get_crops_data()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(crops).encode("utf-8"))
            
        # 3. Serve Dynamic YOLO Person Detection for a Specific Image
        elif path == "/api/detect":
            # Parse query parameters
            query_components = urllib.parse.parse_qs(parsed_url.query)
            filename_list = query_components.get('filename')
            if not filename_list:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Missing filename'}).encode('utf-8'))
                return
                
            filename = filename_list[0]
            filepath = os.path.join(CROPS_DIR, filename)
            
            # Simple directory traversal prevention
            normalized_path = os.path.abspath(filepath)
            expected_prefix = os.path.abspath(CROPS_DIR)
            
            if not (normalized_path.startswith(expected_prefix) and os.path.exists(filepath)):
                self.send_response(404)
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'File not found'}).encode('utf-8'))
                return
                
            # Perform detection or fetch from cache
            if filename in DETECTION_CACHE:
                detections = DETECTION_CACHE[filename]
            else:
                try:
                    frame = cv2.imread(filepath)
                    if frame is not None:
                        detections = get_detector().detect(frame)
                    else:
                        detections = []
                    DETECTION_CACHE[filename] = detections
                except Exception as e:
                    print(f"[Error] Failed to detect person in {filename}: {e}")
                    detections = []
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({'detections': detections}).encode("utf-8"))
            
        # 4. Serve Individual Image Crop Files
        elif path.startswith("/crops/"):
            filename = urllib.parse.unquote(path[len("/crops/"):])
            filepath = os.path.join(CROPS_DIR, filename)
            
            # Simple directory traversal prevention
            normalized_path = os.path.abspath(filepath)
            expected_prefix = os.path.abspath(CROPS_DIR)
            
            if normalized_path.startswith(expected_prefix) and os.path.exists(filepath):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Crop image not found")
        else:
            self.send_error(404, "Route not found")

    def do_POST(self):
        if self.path == "/api/delete":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                filename = data.get('filename')
                if not filename:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(json.dumps({'success': False, 'message': 'Missing filename'}).encode('utf-8'))
                    return
                
                success, message = delete_crop(filename)
                self.send_response(200 if success else 500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({'success': success, 'message': message}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'message': str(e)}).encode('utf-8'))
        else:
            self.send_error(404)

def run_server(port=8000):
    # Warm up YOLO model on main thread
    try:
        print("Warming up YOLOv8n detector weights...")
        get_detector()
    except Exception as e:
        print(f"[Warning] Failed to warm up YOLO detector: {e}")

    # Search for an available port starting at 8000
    while True:
        try:
            server = HTTPServer(('localhost', port), DatasetInspectorRequestHandler)
            break
        except socket.error:
            port += 1
            if port > 8050:
                print("[Error] Could not find an open port between 8000 and 8050.")
                sys.exit(1)
                
    print(f"\n========================================================")
    print(f"NAVIGO Owner Dataset Inspector Tool (Phase 3A)")
    print(f"========================================================")
    print(f"Server is running at: http://localhost:{port}")
    print(f"Press Ctrl+C in this terminal window to stop the server.")
    print(f"========================================================\n")
    
    # Automatically open the web browser in a background thread
    threading.Thread(target=lambda: webbrowser.open(f"http://localhost:{port}")).start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local dataset inspector server. Goodbye!")
        server.server_close()
        sys.exit(0)

# Embedded premium HTML Template with glassmorphism and spring/neon green color scheme
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NAVIGO | Owner Dataset Inspector</title>
    <!-- Outfit Google Font -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0d0e12;
            --panel-color: rgba(22, 25, 35, 0.7);
            --card-color: rgba(28, 32, 46, 0.5);
            --border-color: rgba(255, 255, 255, 0.08);
            --neon-green: #39ff14;
            --neon-green-glow: rgba(57, 255, 20, 0.2);
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
            --danger-red: #ef4444;
            --danger-red-glow: rgba(239, 68, 68, 0.3);
            --font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: var(--font-family);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(at 10% 20%, rgba(57, 255, 20, 0.03) 0px, transparent 50%),
                radial-gradient(at 90% 80%, rgba(57, 255, 20, 0.02) 0px, transparent 50%);
            background-attachment: fixed;
        }

        header {
            background: rgba(13, 14, 18, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            padding: 1.25rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .logo-dot {
            width: 12px;
            height: 12px;
            background-color: var(--neon-green);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--neon-green);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.2); opacity: 0.6; }
            100% { transform: scale(1); opacity: 1; }
        }

        h1 {
            font-size: 1.4rem;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        h1 span {
            color: var(--neon-green);
            text-shadow: 0 0 8px var(--neon-green-glow);
        }

        .stats {
            display: flex;
            gap: 1.5rem;
        }

        .stat-badge {
            background: var(--panel-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.4rem 0.8rem;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .stat-badge strong {
            color: var(--neon-green);
        }

        .controls {
            display: flex;
            gap: 10px;
        }

        .btn {
            background: transparent;
            color: var(--text-color);
            border: 1px solid var(--border-color);
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            font-family: var(--font-family);
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn:hover {
            background: rgba(255, 255, 255, 0.05);
            border-color: var(--text-muted);
        }

        .btn-active {
            border-color: var(--neon-green);
            box-shadow: 0 0 10px var(--neon-green-glow);
            color: var(--neon-green);
        }

        .btn-danger {
            background: rgba(239, 68, 68, 0.1);
            border-color: var(--danger-red);
            color: #ff8b8b;
        }

        .btn-danger:hover {
            background: var(--danger-red);
            color: white;
            box-shadow: 0 0 15px var(--danger-red-glow);
        }

        main {
            flex: 1;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
        }

        /* Views */
        .view-panel {
            display: none;
            width: 100%;
        }

        .view-panel.active {
            display: block;
        }

        /* Grid View */
        .filter-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            gap: 1rem;
        }

        .search-input {
            flex: 1;
            max-width: 400px;
            background: var(--panel-color);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            padding: 0.65rem 1rem;
            border-radius: 8px;
            font-family: var(--font-family);
            outline: none;
            transition: border-color 0.25s;
        }

        .search-input:focus {
            border-color: var(--neon-green);
        }

        .grid-layout {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 1.5rem;
        }

        .crop-card {
            background: var(--card-color);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }

        .crop-card:hover {
            transform: translateY(-4px);
            border-color: rgba(57, 255, 20, 0.3);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }

        .image-container {
            width: 100%;
            padding-top: 56.25%; /* 16:9 Aspect Ratio */
            position: relative;
            background-color: #050608;
            cursor: pointer;
            overflow: hidden;
        }

        .image-container img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.4s;
        }

        .crop-card:hover .image-container img {
            transform: scale(1.05);
        }

        .card-details {
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 6px;
            flex: 1;
            justify-content: space-between;
        }

        .filename-text {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-color);
            word-break: break-all;
            margin-bottom: 4px;
        }

        .meta-line {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .meta-line span.value {
            color: var(--text-color);
            font-weight: 500;
        }

        .card-actions {
            border-top: 1px solid var(--border-color);
            padding: 0.75rem 1rem;
            display: flex;
            justify-content: flex-end;
        }

        /* Slideshow / Inspector Mode */
        .inspector-container {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 2rem;
            background: var(--panel-color);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
            min-height: 500px;
        }

        .viewer-pane {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: #06070a;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            position: relative;
            overflow: hidden;
            padding: 1.5rem;
        }

        .viewer-image-container {
            position: relative;
            max-width: 100%;
            display: inline-block;
        }

        .viewer-image {
            max-width: 100%;
            max-height: 500px;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            display: block;
        }

        .viewer-nav {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: rgba(22, 25, 35, 0.8);
            border: 1px solid var(--border-color);
            color: white;
            font-size: 1.5rem;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            transition: all 0.2s;
            user-select: none;
            z-index: 10;
        }

        .viewer-nav:hover {
            background: var(--neon-green);
            color: black;
            box-shadow: 0 0 15px var(--neon-green-glow);
            border-color: var(--neon-green);
        }

        .viewer-nav-left { left: 15px; }
        .viewer-nav-right { right: 15px; }

        .info-pane {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 0.5rem;
        }

        .info-header h2 {
            font-size: 1.25rem;
            margin-bottom: 1.5rem;
            color: var(--text-color);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.75rem;
        }

        .info-list {
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
            margin-bottom: 2rem;
        }

        .info-item {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .info-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
        }

        .info-val {
            font-size: 1.05rem;
            font-weight: 600;
        }

        .info-val.green {
            color: var(--neon-green);
            text-shadow: 0 0 5px var(--neon-green-glow);
        }

        .kbd-guide {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 1rem;
        }

        .kbd-guide strong {
            color: var(--text-color);
        }
        
        .kbd-key {
            background: #252835;
            border: 1px solid #3c4055;
            border-radius: 4px;
            padding: 1px 6px;
            color: var(--text-color);
            font-size: 0.75rem;
            font-weight: bold;
        }

        .cleaning-guide {
            background: rgba(239, 68, 68, 0.05);
            border: 1px solid rgba(239, 68, 68, 0.15);
            border-radius: 8px;
            padding: 0.8rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 1rem;
        }

        .cleaning-guide strong {
            color: #ff8b8b;
            display: block;
            margin-bottom: 0.4rem;
        }

        .cleaning-guide ul {
            list-style-type: none;
            padding-left: 0;
        }

        .cleaning-guide li {
            margin-bottom: 0.25rem;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            background: var(--panel-color);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 1rem;
        }

        .empty-state h3 {
            font-size: 1.5rem;
            color: var(--text-color);
        }

        .empty-state p {
            color: var(--text-muted);
            max-width: 450px;
            line-height: 1.5;
        }

        /* Toast notifications */
        .toast-container {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .toast {
            background: rgba(22, 25, 35, 0.9);
            border-left: 4px solid var(--neon-green);
            border-top: 1px solid var(--border-color);
            border-bottom: 1px solid var(--border-color);
            border-right: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.8rem 1.5rem;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            display: flex;
            align-items: center;
            gap: 12px;
            animation: slideIn 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
            font-size: 0.9rem;
        }

        .toast.toast-error {
            border-left-color: var(--danger-red);
        }

        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        @keyframes fadeOut {
            to { opacity: 0; transform: translateY(10px); }
        }

        /* Responsive */
        @media (max-width: 900px) {
            .inspector-container {
                grid-template-columns: 1fr;
            }
            header {
                flex-direction: column;
                gap: 1rem;
                padding: 1rem;
            }
            .stats {
                width: 100%;
                justify-content: space-around;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-container">
            <div class="logo-dot"></div>
            <h1>NAVIGO <span>Inspector</span></h1>
        </div>
        <div class="stats">
            <div class="stat-badge">
                <span>Total Frames:</span>
                <strong id="stat-total-crops">0</strong>
            </div>
            <div class="stat-badge">
                <span>YOLO Confidence:</span>
                <strong>&ge; 0.5</strong>
            </div>
        </div>
        <div class="controls">
            <button id="btn-grid-view" class="btn btn-active" onclick="switchView('grid')">
                Grid Gallery
            </button>
            <button id="btn-slide-view" class="btn" onclick="switchView('slide')">
                Inspector Slideshow
            </button>
        </div>
    </header>

    <main>
        <!-- Loading Indicator -->
        <div id="loading" style="text-align: center; padding: 4rem;">
            <h2>Loading dataset...</h2>
        </div>

        <!-- Empty State -->
        <div id="empty-state" class="empty-state" style="display: none;">
            <h3>No crops found</h3>
            <p>Your <code>data/owner</code> folder is currently empty. Please put video files inside <code>data/raw/owner</code> and run the dataset extractor script first.</p>
        </div>

        <!-- Grid View Panel -->
        <div id="view-grid" class="view-panel">
            <div class="filter-bar">
                <input type="text" id="search-bar" class="search-input" placeholder="Filter by source video name..." oninput="filterCrops()">
                <div class="controls">
                    <button class="btn" onclick="randomInspect()">Inspect Random</button>
                </div>
            </div>
            <!-- Grid Cleaning Alert Box -->
            <div class="cleaning-guide" style="margin-bottom: 1.5rem; margin-top: 0; border-color: rgba(57, 255, 20, 0.15); background: rgba(57, 255, 20, 0.03);">
                <strong style="color: var(--neon-green);">Dataset Quality Checklist:</strong>
                <p style="font-size: 0.85rem; line-height: 1.4; color: var(--text-color);">
                    Review the crops below or switch to <strong>Inspector Slideshow</strong>. Manually delete samples that are:
                    heavily cut off, partial body only, blurry, false positive YOLO detections, extremely close/partial crops, or barely visible.
                </p>
            </div>
            <div class="grid-layout" id="grid-container">
                <!-- Cards injected here -->
            </div>
        </div>

        <!-- Slideshow/Inspector Panel -->
        <div id="view-slide" class="view-panel">
            <div class="inspector-container">
                <div class="viewer-pane">
                    <button class="viewer-nav viewer-nav-left" onclick="prevSlide()">&lsaquo;</button>
                    <div class="viewer-image-container">
                        <img id="viewer-img-el" class="viewer-image" src="" alt="Owner Crop">
                    </div>
                    <button class="viewer-nav viewer-nav-right" onclick="nextSlide()">&rsaquo;</button>
                </div>
                <div class="info-pane">
                    <div class="info-header">
                        <h2 id="viewer-filename">filename.jpg</h2>
                        <div class="info-list">
                            <div class="info-item">
                                <span class="info-label">Source Video</span>
                                <span class="info-val" id="viewer-video">video.mp4</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Frame &amp; Time</span>
                                <span class="info-val" id="viewer-frame">Frame 0 | 0.0s</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Saved YOLO Confidence</span>
                                <span class="info-val" id="viewer-saved-conf">N/A</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">YOLO Verification</span>
                                <span class="info-val" id="viewer-conf">Running YOLO...</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Bounding Box(es)</span>
                                <span class="info-val" id="viewer-bbox">Calculating...</span>
                            </div>
                        </div>
                    </div>
                    <div>
                        <button class="btn btn-danger" style="width: 100%; justify-content: center; padding: 0.8rem;" onclick="deleteCurrentSlide()">
                            Delete Crop
                        </button>
                        <div class="kbd-guide">
                            <strong>Keyboard Shortcuts:</strong><br>
                            <span class="kbd-key">&larr;</span> / <span class="kbd-key">&rarr;</span> Navigate Crops<br>
                            <span class="kbd-key">D</span> / <span class="kbd-key">Del</span> Delete Crop<br>
                            <span class="kbd-key">Space</span> Random Crop
                        </div>
                        <div class="cleaning-guide">
                            <strong>Manual Cleaning Criteria:</strong>
                            <ul>
                                <li>❌ Person heavily cut off by frame</li>
                                <li>❌ Only small portion of body visible</li>
                                <li>❌ Severe motion blur</li>
                                <li>❌ Incorrect YOLO detection</li>
                                <li>❌ Extremely close/partial crops</li>
                                <li>❌ Person is barely visible/obscured</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <div class="toast-container" id="toast-container"></div>

    <script>
        let crops = [];
        let filteredCrops = [];
        let currentIndex = 0;
        let currentView = 'grid';
        let currentDetections = [];

        // Load data on start
        window.addEventListener('DOMContentLoaded', fetchCrops);

        // Keyboard listeners for Slide Show
        window.addEventListener('keydown', (e) => {
            if (currentView !== 'slide') return;
            if (e.key === 'ArrowLeft') {
                prevSlide();
            } else if (e.key === 'ArrowRight') {
                nextSlide();
            } else if (e.key === 'Delete' || e.key.toLowerCase() === 'd') {
                deleteCurrentSlide();
            } else if (e.key === ' ') {
                e.preventDefault();
                goToRandom();
            }
        });

        // Window resize listener to keep bounding boxes responsive
        window.addEventListener('resize', () => {
            if (currentView === 'slide' && currentDetections && currentDetections.length > 0) {
                const imgEl = document.getElementById('viewer-img-el');
                drawBoxes(currentDetections, imgEl);
            }
        });

        async function fetchCrops() {
            document.getElementById('loading').style.display = 'block';
            try {
                const response = await fetch('/api/crops');
                crops = await response.json();
                filteredCrops = [...crops];
                
                document.getElementById('loading').style.display = 'none';
                
                updateStats();
                if (crops.length === 0) {
                    document.getElementById('empty-state').style.display = 'flex';
                    document.getElementById('view-grid').style.display = 'none';
                    document.getElementById('view-slide').style.display = 'none';
                } else {
                    document.getElementById('empty-state').style.display = 'none';
                    renderGrid();
                    updateSlideView();
                    switchView(currentView);
                }
            } catch (err) {
                console.error(err);
                showToast("Failed to load crops data.", true);
            }
        }

        function updateStats() {
            document.getElementById('stat-total-crops').textContent = crops.length;
        }

        function switchView(view) {
            currentView = view;
            document.getElementById('btn-grid-view').classList.remove('btn-active');
            document.getElementById('btn-slide-view').classList.remove('btn-active');
            
            document.getElementById('view-grid').classList.remove('active');
            document.getElementById('view-slide').classList.remove('active');

            if (view === 'grid') {
                document.getElementById('btn-grid-view').classList.add('btn-active');
                document.getElementById('view-grid').classList.add('active');
            } else {
                document.getElementById('btn-slide-view').classList.add('btn-active');
                document.getElementById('view-slide').classList.add('active');
                updateSlideView();
            }
        }

        function renderGrid() {
            const container = document.getElementById('grid-container');
            container.innerHTML = '';
            
            filteredCrops.forEach((crop, idx) => {
                const card = document.createElement('div');
                card.className = 'crop-card';
                card.innerHTML = `
                    <div class="image-container" onclick="openInSlideshow('${crop.filename}')">
                        <img src="/crops/${encodeURIComponent(crop.filename)}" loading="lazy" alt="Crop">
                    </div>
                    <div class="card-details">
                        <div class="filename-text" title="${crop.filename}">${crop.filename}</div>
                        <div class="meta-line">
                            <span>Video:</span>
                            <span class="value" title="${crop.source_video}">${crop.source_video.length > 18 ? crop.source_video.substring(0, 15) + '...' : crop.source_video}</span>
                        </div>
                    </div>
                    <div class="card-actions">
                        <button class="btn btn-danger" style="padding: 0.35rem 0.75rem; font-size: 0.8rem;" onclick="deleteCropItem('${crop.filename}')">
                            Delete
                        </button>
                    </div>
                `;
                container.appendChild(card);
            });
        }

        function filterCrops() {
            const query = document.getElementById('search-bar').value.toLowerCase();
            filteredCrops = crops.filter(crop => 
                crop.source_video.toLowerCase().includes(query) || 
                crop.filename.toLowerCase().includes(query)
            );
            renderGrid();
        }

        function openInSlideshow(filename) {
            const idx = filteredCrops.findIndex(c => c.filename === filename);
            if (idx !== -1) {
                currentIndex = idx;
                switchView('slide');
            }
        }

        function clearOverlay() {
            const overlay = document.querySelector('.bbox-overlay');
            if (overlay) {
                overlay.remove();
            }
            currentDetections = [];
        }

        function drawBoxes(detections, imgEl) {
            clearOverlay();
            if (!detections || detections.length === 0) return;
            
            // Layout dimensions
            const container = imgEl.parentElement;
            
            const canvas = document.createElement('canvas');
            canvas.className = 'bbox-overlay';
            canvas.style.position = 'absolute';
            canvas.style.top = imgEl.offsetTop + 'px';
            canvas.style.left = imgEl.offsetLeft + 'px';
            canvas.style.width = imgEl.clientWidth + 'px';
            canvas.style.height = imgEl.clientHeight + 'px';
            canvas.width = imgEl.clientWidth;
            canvas.height = imgEl.clientHeight;
            canvas.style.pointerEvents = 'none';
            canvas.style.zIndex = '5';
            
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const scaleX = imgEl.clientWidth / imgEl.naturalWidth;
            const scaleY = imgEl.clientHeight / imgEl.naturalHeight;
            
            detections.forEach(det => {
                const [x1, y1, x2, y2] = det.box;
                const confidence = det.confidence;
                
                const rx1 = x1 * scaleX;
                const ry1 = y1 * scaleY;
                const rx2 = x2 * scaleX;
                const ry2 = y2 * scaleY;
                const rw = rx2 - rx1;
                const rh = ry2 - ry1;
                
                // Draw Box border
                ctx.strokeStyle = '#39ff14'; // Neon green
                ctx.lineWidth = 2;
                ctx.strokeRect(rx1, ry1, rw, rh);
                
                // Draw Shadow / Glow
                ctx.strokeStyle = 'rgba(57, 255, 20, 0.4)';
                ctx.lineWidth = 6;
                ctx.strokeRect(rx1, ry1, rw, rh);
                
                // Draw Text Banner
                const label = `Person: ${(confidence * 100).toFixed(1)}%`;
                ctx.font = 'bold 11px Outfit, sans-serif';
                const textWidth = ctx.measureText(label).width;
                
                ctx.fillStyle = '#39ff14';
                ctx.fillRect(rx1, ry1 - 18, textWidth + 10, 18);
                
                ctx.fillStyle = '#000000';
                ctx.fillText(label, rx1 + 5, ry1 - 5);
            });
            
            container.appendChild(canvas);
        }

        function updateSlideView() {
            if (filteredCrops.length === 0) {
                switchView('grid');
                return;
            }
            if (currentIndex >= filteredCrops.length) {
                currentIndex = filteredCrops.length - 1;
            }
            if (currentIndex < 0) {
                currentIndex = 0;
            }

            const crop = filteredCrops[currentIndex];
            const filename = crop.filename;
            
            clearOverlay();
            
            document.getElementById('viewer-filename').textContent = filename;
            document.getElementById('viewer-video').textContent = crop.source_video;
            document.getElementById('viewer-frame').textContent = `Frame ${crop.frame_number} | Time: ${crop.timestamp_seconds}s`;
            document.getElementById('viewer-saved-conf').textContent = crop.yolo_confidence || 'N/A';
            
            document.getElementById('viewer-conf').textContent = "Running YOLOv8n Verification...";
            document.getElementById('viewer-conf').style.color = "var(--text-color)";
            document.getElementById('viewer-bbox').textContent = "Calculating bounding boxes...";
            
            const imgEl = document.getElementById('viewer-img-el');
            imgEl.src = `/crops/${encodeURIComponent(filename)}`;
            
            imgEl.onload = async () => {
                // Ensure slide hasn't changed during network fetch
                if (filteredCrops[currentIndex].filename !== filename) return;
                
                try {
                    const response = await fetch(`/api/detect?filename=${encodeURIComponent(filename)}`);
                    const data = await response.json();
                    
                    // Double check slide again
                    if (filteredCrops[currentIndex].filename !== filename) return;
                    
                    currentDetections = data.detections;
                    drawBoxes(currentDetections, imgEl);
                    
                    const count = currentDetections.length;
                    if (count === 0) {
                        document.getElementById('viewer-conf').innerHTML = '<span style="color: #ef4444; font-weight: bold;">NO PERSON DETECTED</span><br><span style="font-size: 0.95rem; color: var(--text-muted);">Confidence: N/A</span>';
                        document.getElementById('viewer-bbox').textContent = "N/A";
                    } else if (count === 1) {
                        const confVal = currentDetections[0].confidence.toFixed(2);
                        document.getElementById('viewer-conf').innerHTML = '<span style="color: var(--neon-green); font-weight: bold;">Person detected</span><br><span style="font-size: 0.95rem; color: var(--text-color);">Confidence: ' + confVal + '</span>';
                        
                        const bbox = currentDetections[0].box;
                        document.getElementById('viewer-bbox').textContent = `[${bbox.join(', ')}]`;
                    } else {
                        const confTexts = currentDetections.map(d => d.confidence.toFixed(2)).join(', ');
                        document.getElementById('viewer-conf').innerHTML = '<span style="color: var(--neon-green); font-weight: bold;">Multiple people detected</span><br><span style="font-size: 0.95rem; color: var(--text-color);">Confidence: ' + confTexts + '</span>';
                        
                        const bboxes = currentDetections.map(d => `[${d.box.join(', ')}]`).join('<br>');
                        document.getElementById('viewer-bbox').innerHTML = bboxes;
                    }
                } catch (err) {
                    console.error(err);
                    document.getElementById('viewer-conf').textContent = "Detection Failed";
                    document.getElementById('viewer-bbox').textContent = "Error running YOLOv8n";
                }
            };
        }

        function prevSlide() {
            if (filteredCrops.length <= 1) return;
            currentIndex = (currentIndex - 1 + filteredCrops.length) % filteredCrops.length;
            updateSlideView();
        }

        function nextSlide() {
            if (filteredCrops.length <= 1) return;
            currentIndex = (currentIndex + 1) % filteredCrops.length;
            updateSlideView();
        }

        function goToRandom() {
            if (filteredCrops.length <= 1) return;
            let newIdx = currentIndex;
            while (newIdx === currentIndex) {
                newIdx = Math.floor(Math.random() * filteredCrops.length);
            }
            currentIndex = newIdx;
            updateSlideView();
        }

        function randomInspect() {
            if (crops.length === 0) return;
            filteredCrops = [...crops];
            document.getElementById('search-bar').value = '';
            renderGrid();
            currentIndex = Math.floor(Math.random() * filteredCrops.length);
            switchView('slide');
        }

        async function deleteCropItem(filename) {
            if (!confirm(`Are you sure you want to delete frame ${filename}? This action will delete the original image file and remove it from metadata CSV.`)) {
                return;
            }
            
            try {
                const response = await fetch('/api/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename })
                });
                const resData = await response.json();
                
                if (resData.success) {
                    showToast("Deleted " + filename);
                    crops = crops.filter(c => c.filename !== filename);
                    filteredCrops = filteredCrops.filter(c => c.filename !== filename);
                    
                    updateStats();
                    if (crops.length === 0) {
                        fetchCrops();
                    } else {
                        renderGrid();
                        updateSlideView();
                    }
                } else {
                    showToast("Error: " + resData.message, true);
                }
            } catch (err) {
                console.error(err);
                showToast("System error deleting crop.", true);
            }
        }

        async function deleteCurrentSlide() {
            if (filteredCrops.length === 0) return;
            const filename = filteredCrops[currentIndex].filename;
            
            try {
                const response = await fetch('/api/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename })
                });
                const resData = await response.json();
                
                if (resData.success) {
                    showToast("Deleted " + filename);
                    crops = crops.filter(c => c.filename !== filename);
                    filteredCrops = filteredCrops.filter(c => c.filename !== filename);
                    
                    updateStats();
                    if (crops.length === 0) {
                        fetchCrops();
                    } else {
                        if (currentIndex >= filteredCrops.length) {
                            currentIndex = filteredCrops.length - 1;
                        }
                        updateSlideView();
                        renderGrid();
                    }
                } else {
                    showToast("Error: " + resData.message, true);
                }
            } catch (err) {
                console.error(err);
                showToast("System error deleting crop.", true);
            }
        }

        function showToast(message, isError = false) {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = 'toast' + (isError ? ' toast-error' : '');
            toast.textContent = message;
            container.appendChild(toast);
            
            setTimeout(() => {
                toast.style.animation = 'fadeOut 0.4s forwards';
                setTimeout(() => toast.remove(), 400);
            }, 3000);
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    run_server()
