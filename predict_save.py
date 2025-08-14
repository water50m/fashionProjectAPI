import cv2
import os
import csv
import time
from ultralytics import YOLO
from config import load_config,save_config
import re


#
# ===============================
# CONFIGURATION
# ===============================
config=load_config()
VIDEO_FOLDERS = [
    config["VIDEO_PATH"]
]
MODEL_PATH = config["AI_MODEL_PATH"]+config["AI_MODEL_NAME"]
OUTPUT_CSV = config["CSV_R_PATH"]
PROCESSED_LOG = config["PROCESSED_LOG"]

CLASS_NAMES = [
    'short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear',
    'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress',
    'long sleeve dress', 'vest dress', 'sling dress'
]
IOU_THRESHOLD = config["MODEL_CONFIG"]["iou_threshold"]
CONFIDENCE_THRESHOLD = config["MODEL_CONFIG"]["confidence_threshold"]
FREQUENCY = config["MODEL_CONFIG"]["frequency"]
DETECTING_ALL = config["MODEL_CONFIG"]["detectingAll"]
DETECTION_RESULT_PATH = config["RESULTS_PREDICT_DIR"]
DETECTION_RESULT_FILE = config["DETECTION_RESULT_FILE"]
# ===============================
# FUNCTION: Load model
# ===============================
def load_model(model_path):
    start = time.time()
    model = YOLO(model_path)
    print(f"[load_model] ใช้เวลา: {time.time() - start:.4f} วินาที")
    return model

# ===============================
# FUNCTION: Load list of already processed files
# ===============================
def load_processed_files(log_path):
    start = time.time()
    if not os.path.exists(log_path):
        print(f"[load_processed_files] ไม่มีไฟล์ log ใช้เวลา: {time.time() - start:.4f} วินาที")
        return set()
    with open(log_path, "r") as f:
        processed = set(line.strip() for line in f)
    print(f"[load_processed_files] โหลด {len(processed)} ไฟล์ ใช้เวลา: {time.time() - start:.4f} วินาที")
    return processed

# ===============================
# FUNCTION: Save processed file name
# ===============================
def mark_file_as_processed(log_path, filename):
    with open(log_path, "a") as f:
        f.write(filename + "\n")
        
    print(f"[mark_file_as_processed] บันทึกไฟล์: {filename}")

def clear_processed_files(log_path):
    with open(log_path, "w") as f:
        f.truncate()
    print(f"[clear_processed_files] เคลียร์ไฟล์ log")

# ===============================
# FUNCTION: Get all video paths from multiple folders
# ===============================
def get_video_files(folders):
    start = time.time()
    video_files = []
    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    video_files.append(os.path.join(root, file))
    print(f"[get_video_files] พบ {len(video_files)} ไฟล์ ใช้เวลา: {time.time() - start:.4f} วินาที")
    return video_files

# ===============================
# FUNCTION: Detect objects in video
# ===============================
def detect_objects_in_video(model, video_path,confidence_threshold,frequency):
    start = time.time()
    detections = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    frame_interval = int(fps * frequency)  # ตรวจจับทุก 0.1 วินาที

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # ตรวจเฉพาะเมื่อถึงช่วงที่กำหนด
        if frame_index % frame_interval == 0:
            results = model.track(source=frame, persist=True, verbose=False)
            boxes = results[0].boxes

            for box in boxes:
                conf = float(box.conf[0])
                if conf < confidence_threshold:
                    continue  # ข้าม box นี้
                
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                timestamp = frame_index / fps

                detections.append({
                    "filename": os.path.basename(video_path),
                    "timestamp": round(timestamp, 2),
                    "class": cls,
                    "class_name": CLASS_NAMES[cls],
                    "confidence": round(conf, 2),
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "track_id": track_id
                })

        frame_index += 1

    cap.release()
    print(f"[detect_objects_in_video] {os.path.basename(video_path)}: ตรวจพบ {len(detections)} รายการ ใช้เวลา: {time.time() - start:.2f} วินาที")
    return detections


# ===============================
# FUNCTION: Save detection results to CSV
# ===============================
def create_csv_file(csv_floder_path):
    csv_files = [f for f in os.listdir(csv_floder_path) if f.endswith('.csv')]
    numbers = []
    pattern = re.compile(r"detection_results_(\d+)\.csv")
    for file in csv_files:
        match = pattern.match(file)
        if match:
            numbers.append(int(match.group(1)))

    next_number = max(numbers) + 1 if numbers else 1
    new_filename = f"detection_results_{next_number}.csv"
    new_filepath = os.path.join(csv_floder_path, new_filename)
    
    with open(new_filepath, "w", newline="") as csvfile:
        fieldnames = [
            "filename", "timestamp", "class", "class_name", "confidence",
            "x", "y", "w", "h", "track_id"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    save_config({"DETECTION_RESULT_FILE": new_filename})
    save_config({"CSV_R_PATH": new_filepath})
    return new_filepath
    
def save_detections_to_csv(csv_path, detections):
    start = time.time()
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        fieldnames = [
            "filename", "timestamp", "class", "class_name", "confidence",
            "x", "y", "w", "h", "track_id"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in detections:
            writer.writerow(row)
    print(f"[save_detections_to_csv] เขียนข้อมูล {len(detections)} แถว ใช้เวลา: {time.time() - start:.4f} วินาที")

# ===============================
# MAIN SCRIPT
# ===============================
def detection(configuration):
    print("start detection")
    if not configuration.auto_config:
        MODEL_PATH = configuration.custom_ai_path
        FREQUENCY = configuration.frequency
        CONFIDENCE_THRESHOLD = configuration.confidence
        DETECTING_ALL = configuration.detectingAll

    if DETECTING_ALL:
        clear_processed_files(PROCESSED_LOG)
        OUTPUT_CSV = create_csv_file(DETECTION_RESULT_PATH)
    else:
        OUTPUT_CSV = config["CSV_R_PATH"]
        
    total_start = time.time()

    model = load_model(MODEL_PATH)
    processed = load_processed_files(PROCESSED_LOG)
    video_files = get_video_files(VIDEO_FOLDERS)

    for video_path in video_files:
        filename = os.path.basename(video_path)
        # ถ้า detect แล้ว และ ไม่ต้องตรวจจับทั้งหมด ให้เข้าเงื่อนไขนี้
        if (filename in processed) and not DETECTING_ALL:
            print(f"[✔] Already processed: {filename}")
            continue

        print(f"[▶] Processing: {filename}")
        detections = detect_objects_in_video(model, video_path,CONFIDENCE_THRESHOLD,FREQUENCY)

        # if detections:
            
        #     save_detections_to_csv(OUTPUT_CSV, detections)

        # mark_file_as_processed(PROCESSED_LOG, filename,DETECTING_ALL)

    print(f"[DONE] ทั้งหมดใช้เวลา: {time.time() - total_start:.2f} วินาที")

