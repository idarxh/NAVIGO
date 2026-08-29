import os
import cv2
import csv
import sys
import argparse
from pathlib import Path
import numpy as np

# Import the PersonDetector from NAVIGO source files
try:
    from src.detection import PersonDetector
except ModuleNotFoundError:
    from detection import PersonDetector

def check_and_clear_dataset(output_dir, csv_path, json_path=None, cls="owner"):
    """
    Checks if there are existing crops in output_dir or an existing metadata CSV.
    If so, prints details of what will be deleted and prompts the user for confirmation.
    If confirmed, deletes them. If not, exits.
    """
    output_path = Path(output_dir)
    csv_path = Path(csv_path)
    
    existing_crops = []
    if output_path.exists():
        existing_crops = [f for f in output_path.iterdir() if f.is_file() and f.name != '.gitkeep']
        
    csv_exists = csv_path.exists()
    
    if existing_crops or csv_exists:
        print("\n[Warning] Previously generated crops/metadata found:")
        if existing_crops:
            print(f"  - {len(existing_crops)} crop image files in '{output_path}'")
        if csv_exists:
            print(f"  - Metadata CSV file at '{csv_path}'")
            
        print("\nThis will perform a COMPLETE FRESH DATASET EXTRACTION.")
        print(f"To proceed, all previously generated crops and metadata inside data/{cls}/ will be deleted.")
        print(f"Note: The original videos inside data/raw/{cls}/ WILL NOT be modified or deleted.")
        
        try:
            # Interactive confirmation
            response = input("\nDo you confirm the deletion of previous crops/metadata? (yes/no): ").strip().lower()
        except KeyboardInterrupt:
            print("\n[Info] Interrupted by user. Exiting.")
            sys.exit(0)
            
        if response not in ('yes', 'y', 'confirm'):
            print("\n[Info] Extraction cancelled by user. Exiting.")
            sys.exit(0)
            
        print("\n[Info] Deleting previously generated dataset crops/metadata...")
        # Delete crop images
        for f in existing_crops:
            try:
                f.unlink()
            except Exception as e:
                print(f"  [Error] Could not delete {f}: {e}")
                
        # Delete CSV metadata file
        if csv_exists:
            try:
                csv_path.unlink()
            except Exception as e:
                print(f"  [Error] Could not delete metadata CSV: {e}")
                
        # Delete JSON processing log if exists
        if json_path and os.path.exists(json_path):
            try:
                os.remove(json_path)
            except Exception as e:
                print(f"  [Error] Could not delete processing log JSON: {e}")
                
        print("[Info] Clean-up complete. Starting fresh dataset extraction.\n")

def crop_and_pad_person(frame, box, padding_ratio=0.1, target_size=(224, 224)):
    """
    Crops the detected person from the frame with a configurable padding,
    preserving the aspect ratio and applying grey letterboxing/padding to match target_size.
    """
    h_orig, w_orig, _ = frame.shape
    x1, y1, x2, y2 = box
    
    bw = x2 - x1
    bh = y2 - y1
    
    # Calculate padding in pixels
    pad_w = int(bw * padding_ratio)
    pad_h = int(bh * padding_ratio)
    
    # Apply padding and constrain to image boundaries
    x1_pad = max(0, x1 - pad_w)
    y1_pad = max(0, y1 - pad_h)
    x2_pad = min(w_orig, x2 + pad_w)
    y2_pad = min(h_orig, y2 + pad_h)
    
    # Crop the padded bounding box
    cropped = frame[y1_pad:y2_pad, x1_pad:x2_pad]
    
    # If crop is empty, return None
    if cropped.size == 0:
        return None
        
    crop_h, crop_w, _ = cropped.shape
    target_w, target_h = target_size
    
    # Calculate scale factor to preserve aspect ratio
    scale = min(target_w / crop_w, target_h / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    
    # Ensure dimensions are at least 1 pixel
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    # Resize keeping aspect ratio
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create grey canvas (BGR (128, 128, 128))
    canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    
    # Center the resized image on the canvas
    dx = (target_w - new_w) // 2
    dy = (target_h - new_h) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized
    
    return canvas

def main():
    parser = argparse.ArgumentParser(description="NAVIGO Phase 3A Person-Crop Dataset Extractor")
    parser.add_argument("--class", dest="cls", type=str, default="owner", choices=["owner", "others"], help="Class to extract (owner or others).")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory containing raw videos.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save extracted crops.")
    parser.add_argument("--fps", type=float, default=3.0, help="Approximate frames per second to sample.")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to the metadata CSV file.")
    parser.add_argument("--padding-ratio", type=float, default=0.1, help="Padding ratio around the bounding box (default: 0.1).")
    parser.add_argument("--target-size", type=str, default="224x224", help="Target output image dimensions (default: 224x224).")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold for person detection.")
    parser.add_argument("--log-path", type=str, default=None, help="Path to the old JSON log (which will be deleted if present).")
    
    args = parser.parse_args()
    
    # Resolve paths based on class if not explicitly provided
    if args.video_dir is None:
        args.video_dir = f"data/raw/{args.cls}"
    if args.output_dir is None:
        args.output_dir = f"data/{args.cls}"
    if args.csv_path is None:
        args.csv_path = f"data/{args.cls}_metadata.csv"
    if args.log_path is None:
        args.log_path = f"data/{args.cls}_processed.json"
    
    # Parse target size
    try:
        tw, th = map(int, args.target_size.split('x'))
        target_size = (tw, th)
    except ValueError:
        print(f"[Error] Invalid target size '{args.target_size}'. Must be in format WxH (e.g. 224x224).")
        sys.exit(1)
        
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    csv_path = Path(args.csv_path)
    log_path = Path(args.log_path)
    
    # 1. Interactive Cleanup
    check_and_clear_dataset(output_dir, csv_path, log_path, cls=args.cls)
    
    # Ensure output folders exist
    video_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find supported video files
    supported_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    video_files = [
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in supported_extensions
    ]
    
    if not video_files:
        print(f"\n[Warning] No supported video files found in: {video_dir.resolve()}")
        print(f"Please copy your {args.cls} videos (.mp4, .mov, .avi, or .mkv) to that directory and rerun the script.")
        return
        
    print(f"\n========================================================")
    print(f"NAVIGO PHASE 3A: {args.cls.upper()} Dataset Extractor")
    print(f"========================================================")
    print(f"Source Directory:       {video_dir.resolve()}")
    print(f"Output Directory:       {output_dir.resolve()}")
    print(f"Sampling Rate:          {args.fps} frames per second")
    print(f"YOLO Confidence Thresh: {args.conf}")
    print(f"BBox Padding Ratio:     {args.padding_ratio}")
    print(f"Target Size (WxH):      {args.target_size}")
    print(f"Metadata CSV:           {csv_path}")
    print(f"Total Videos Found:     {len(video_files)}")
    print(f"========================================================\n")
    
    # Initialize detector
    print("Initializing YOLOv8n detector weights...")
    try:
        detector = PersonDetector(model_path="yolov8n.pt", confidence_threshold=args.conf)
    except Exception as e:
        print(f"[Error] Failed to initialize detector: {e}")
        sys.exit(1)
        
    # Open CSV for writing
    csv_headers = [
        'image_filename', 
        'source_video', 
        'frame_number', 
        'timestamp', 
        'timestamp_seconds',
        'original_frame_width', 
        'original_frame_height', 
        'bbox_x1', 
        'bbox_y1', 
        'bbox_x2', 
        'bbox_y2', 
        'yolo_confidence', 
        'number_of_people'
    ]
    
    try:
        csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
        csv_writer.writeheader()
    except Exception as e:
        print(f"[Error] Failed to open CSV file {csv_path} for writing: {e}")
        sys.exit(1)
        
    # Statistics variables
    total_videos_processed = 0
    total_frames_read = 0
    total_frames_sampled = 0
    frames_0_people = 0
    frames_1_person = 0
    frames_2plus_people = 0
    crops_saved = 0
    sum_yolo_confidence = 0.0
    
    try:
        for idx, video_path in enumerate(video_files, 1):
            video_name = video_path.name
            print(f"\n[{idx}/{len(video_files)}] Processing Video: {video_name}")
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"  [Error] Could not open video file: {video_path}")
                continue
                
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if video_fps <= 0:
                video_fps = 30.0  # Fallback
                
            # Determine the step size to achieve the target sampling FPS
            frame_step = max(1, int(round(video_fps / args.fps)))
            
            print(f"  Specs: {video_fps:.2f} FPS | {total_video_frames} total frames | Frame Step: {frame_step}")
            
            frame_idx = 0
            video_sampled = 0
            video_saved = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                total_frames_read += 1
                
                # Sample frames based on frame step
                if frame_idx % frame_step == 0:
                    total_frames_sampled += 1
                    video_sampled += 1
                    timestamp = frame_idx / video_fps
                    
                    # Image dimensions
                    h_orig, w_orig, _ = frame.shape
                    
                    # Detect people
                    detections = detector.detect(frame)
                    num_people = len(detections)
                    
                    if num_people == 0:
                        frames_0_people += 1
                    elif num_people == 1:
                        frames_1_person += 1
                        det = detections[0]
                        box = det['box']
                        conf = det['confidence']
                        
                        # Generate unique filename: source_video_name_frame_XXXXXX.jpg
                        video_stem = video_path.stem
                        image_filename = f"{video_stem}_frame_{frame_idx:06d}.jpg"
                        frame_path = output_dir / image_filename
                        
                        # Crop and apply padding & aspect-ratio letterboxing
                        crop_img = crop_and_pad_person(
                            frame=frame,
                            box=box,
                            padding_ratio=args.padding_ratio,
                            target_size=target_size
                        )
                        
                        if crop_img is not None:
                            # Save the image
                            cv2.imwrite(str(frame_path), crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            crops_saved += 1
                            video_saved += 1
                            sum_yolo_confidence += conf
                            
                            # Write to CSV
                            csv_writer.writerow({
                                'image_filename': image_filename,
                                'source_video': video_name,
                                'frame_number': frame_idx,
                                'timestamp': round(timestamp, 3),
                                'timestamp_seconds': round(timestamp, 3),
                                'original_frame_width': w_orig,
                                'original_frame_height': h_orig,
                                'bbox_x1': box[0],
                                'bbox_y1': box[1],
                                'bbox_x2': box[2],
                                'bbox_y2': box[3],
                                'yolo_confidence': round(conf, 3),
                                'number_of_people': 1
                            })
                            # Flush metadata file immediately to prevent loss
                            csv_file.flush()
                    else:
                        # 2+ people detected
                        frames_2plus_people += 1
                        
                frame_idx += 1
                
                # Print live progress
                if frame_idx % frame_step == 0 or frame_idx == total_video_frames:
                    progress_pct = min(100.0, (frame_idx / total_video_frames) * 100.0) if total_video_frames > 0 else 100.0
                    sys.stdout.write(
                        f"\r  Progress: {progress_pct:5.1f}% | Frames: {frame_idx}/{total_video_frames} | "
                        f"Sampled: {video_sampled} | Saved: {video_saved}"
                    )
                    sys.stdout.flush()
                    
            cap.release()
            print()  # Final newline for progress
            total_videos_processed += 1
            
    except KeyboardInterrupt:
        print("\n\n[Warning] Extraction process interrupted by user.")
    finally:
        csv_file.close()
        
    # Print summary statistics
    avg_conf = sum_yolo_confidence / crops_saved if crops_saved > 0 else 0.0
    
    print(f"\n========================================================")
    print(f"Extraction Process Completed")
    print(f"========================================================")
    print(f"Videos processed: {total_videos_processed}")
    print(f"Total frames read: {total_frames_read}")
    print(f"Total frames sampled: {total_frames_sampled}")
    print(f"0 people: {frames_0_people}")
    print(f"1 person: {frames_1_person}")
    print(f"2+ people: {frames_2plus_people}")
    print(f"Crops saved: {crops_saved}")
    print(f"Average YOLO confidence of saved crops: {avg_conf:.3f}")
    print(f"========================================================\n")

if __name__ == "__main__":
    main()
