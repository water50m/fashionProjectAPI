import os
import cv2
import csv
import json
import signal
import datetime
import multiprocessing
from ultralytics import YOLO
import time
from config import save_config,load_config
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import uuid


# ===============================
# CONFIG
# ===============================
config = load_config()
CLASS_NAMES_B = [
    'short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear',
    'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress',
    'long sleeve dress', 'vest dress', 'sling dress'
]

CONFIG_PATH = "config/config.json"
RESULTS_PREDICT_DIR = "running/detection_results"
LOG_PATH = "running/log/prediction_log.csv"
PROCESSED_LOG = "running/log/processed_files.txt"
AI_MODEL_PATH = "E:\\ALL_CODE\\python\\fashionV1\\fastapi-project\\model\\fashion13cls.pt"
VIDEO_PATH = "E:\\ALL_CODE\\nextjs\\fashion_project_1\\public\\videos"
TRACKING_AI = "E:\\ALL_CODE\\python\\fashionV1\\fastapi-project\\scrripts\\yolo11n.pt"
model_color=YOLO("E:\\ALL_CODE\\nextjs\\fashion_project_1\\scripts\\model\\segmentModel2.pt")


# ===============================
# LOAD CONFIG OR CUSTOM PATH
# ===============================
def load_config_(custom_path=None):
    with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)

            CSV_PEOPLE_DETECTION = config["CSV_PEOPLE_DETECTION"]
            PROCESSED_LOG = config["PROCESSED_LOG"]
            AI_MODEL_PATH = config["AI_MODEL_PATH"]+config["AI_MODEL_NAME"]
            VIDEO_PATH = config["VIDEO_PATH"]
            LOG_PATH = config["LOG_PATH"]
            RESULTS_PREDICT_DIR = config["RESULTS_PREDICT_DIR"]
            TRACKING_AI = config["TRACKING_AI"]
            confidence = config["MODEL_CONFIG"]["confidence_threshold"]
            frequency = config["MODEL_CONFIG"]["frequency"]
            GET_COLOR_MODEL = config["GET_COLOR_MODEL"]
            if custom_path :
                confidence = custom_path.confidence
                frequency = custom_path.frequency
                AI_MODEL_PATH = custom_path.custom_ai_path
                VIDEO_PATH = custom_path.video_path
                if custom_path.use_system_ai:
                    AI_MODEL_PATH = config["AI_MODEL_PATH"]+custom_path.system_model
                return {
                    "GET_COLOR_MODEL": GET_COLOR_MODEL,
                    "CSV_PEOPLE_DETECTION": CSV_PEOPLE_DETECTION,
                    "RESULTS_PREDICT_DIR": RESULTS_PREDICT_DIR,
                    "PROCESSED_LOG": PROCESSED_LOG,
                    "AI_MODEL_PATH": AI_MODEL_PATH,
                    "LOG_PATH": LOG_PATH,
                    "VIDEO_PATH": VIDEO_PATH,
                    "TRACKING_AI": TRACKING_AI,
                    "confidence": confidence,
                    "frequency": frequency}
            else:
                return {"GET_COLOR_MODEL": GET_COLOR_MODEL,
                        "CSV_PEOPLE_DETECTION": CSV_PEOPLE_DETECTION,
                        "PROCESSED_LOG": PROCESSED_LOG,
                        "AI_MODEL_PATH": AI_MODEL_PATH,
                        "VIDEO_PATH": VIDEO_PATH,
                        "LOG_PATH": LOG_PATH,
                        "RESULTS_PREDICT_DIR": RESULTS_PREDICT_DIR,
                        "TRACKING_AI": TRACKING_AI,
                        "confidence": confidence,
                        "frequency": frequency}

# ===============================
# SETUP RESULT CSV
# ===============================
def get_result_csv(dir,detect_all,type_of_detection):
    cfg = load_config_()
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    os.makedirs(dir, exist_ok=True)
    if detect_all:
        # สร้างไฟล์ใหม่ ถ้าวันเดียวกันซ้ำ ให้เพิ่มเลขต่อท้าย
        base_name = os.path.join(dir, f"{type_of_detection}\\results_{type_of_detection}_{date_str}.csv")
        if not os.path.exists(base_name):
            return base_name
        counter = 1
        while True:
            new_name = os.path.join(dir, f"{type_of_detection}\\results_{type_of_detection}_{date_str}_{counter}.csv")
            if not os.path.exists(new_name):
                return new_name
            counter += 1
    else:
        # ใช้ไฟล์ล่าสุด
        files = [f for f in os.listdir(dir) if f.startswith(f"{type_of_detection}\\results_{type_of_detection}_{date_str}")]
        if files:
            return os.path.join(dir, sorted(files)[-1])
        return os.path.join(dir, f"{type_of_detection}\\results_{type_of_detection}_{date_str}.csv")

# ===============================
# LOAD/UPDATE PROCESSED FILES
# ===============================
def load_processed_files():
    if not os.path.exists(PROCESSED_LOG):
        return set()
    with open(PROCESSED_LOG, 'r') as f:
        return set(line.strip() for line in f)

def mark_file_as_processed(filename):
    with open(PROCESSED_LOG, 'a') as f:
        f.write(filename + "\n")

# ===============================
# WRITE LOG
# ===============================
def write_log(filename):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if new_file:
            writer.writerow(["filename", "datetime"])
        writer.writerow([filename, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# ===============================
# GET ALL VIDEOS
# ===============================
def get_video_files(folders):
    video_files = []
    for folder in folders:
        print(folder)
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    video_files.append(os.path.join(root, file))
    print(f"[get_video_files] พบ {len(video_files)} ไฟล์")
    return video_files
# ===============================
# CLEAN DATA
# ===============================
def clean_data(detections):

    df = pd.DataFrame(detections)

    if df.empty:
        return []

    # 1️⃣ หาว่า 2 class ที่มีจำนวน detection เยอะที่สุด
    top_classes = df['class'].value_counts().head(2).index.tolist()

    # 2️⃣ กรองเฉพาะ top 2 class
    df_top = df[df['class'].isin(top_classes)]

    # 3️⃣ เลือก row ที่มี confidence สูงสุดของแต่ละ class
    best_per_class = (
        df_top.loc[df_top.groupby('class')['confidence'].idxmax()]
        .to_dict(orient='records')
    )

    return best_per_class

# ===============================
# GET COLOR 
# ===============================
# ฟังก์ชันสร้างภาพสี่เหลี่ยมสี
def show_color_window(color_bgr, name="Color"):
    swatch = np.zeros((100, 100, 3), dtype=np.uint8)  # สี่เหลี่ยมเปล่า
    swatch[:] = np.uint8(color_bgr)  # ใส่สี
    cv2.imshow(name, swatch)

def get_color1(model, image,width,height):
    result = model.predict(
                source=image, 
                verbose=False
            )[0]

    mean_color_bgr = [0,0,0]

    if result.masks is not None and len(result.masks.data) > 0:
        mask = result.masks.data[0].cpu().numpy()
        mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_resized.astype(bool)
        pixels_in_mask = image[mask_bool]
        # mean color
        mean_color_bgr = pixels_in_mask.mean(axis=0)
        # # median color
        # median_color_bgr = np.median(pixels_in_mask, axis=0)
        # # dominant color
        # pixels = pixels_in_mask.reshape(-1, 3)
        # kmeans = KMeans(n_clusters=1, random_state=0).fit(pixels)
        # dominant_color_bgr = kmeans.cluster_centers_[0]
    return mean_color_bgr  

def get_color_dominant(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_  # RGB
    counts = np.bincount(kmeans.labels_)
    dominant = colors[np.argmax(counts)][::-1]
    return str(dominant)
# ===============================
# DETECTION FUNCTION
# ===============================


def detect_objects(video_path, model_A, model_B,  output_csv, cfg,resultpeople_detection_csv,class_selected=None):

    cap = cv2.VideoCapture(video_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    
    clean_detections_clothing = []
    people_detections = []
    seen_track_ids = set()
    frame_interval = int(fps * cfg["frequency"])  # ตรวจจับทุก cfg["frequency"] วินาที
    filename = os.path.basename(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_index % frame_interval == 0:    
            # Track ด้วย model_A (เฉพาะ class person=0)
            track_results = model_A.track(
                source=frame,
                persist=True,
                classes=[0],
                verbose=False,
                tracker='botsort.yaml'
            )[0]
             
            if track_results.boxes.id is not None:
                for box, tid in zip(track_results.boxes.xyxy, track_results.boxes.id):
                    detections = []
                    x1, y1, x2, y2 = map(int, box.tolist())
                    person_crop = frame[y1:y2, x1:x2]
                    timestamp = round(frame_index / fps, 2)
                    # Predict เสื้อผ้าเฉพาะเมื่อมี track_id ใหม่ 
                    if tid not in seen_track_ids:
                        # seen_track_ids.add(tid) # ถ้าต้องการให้ตรวจครั้งเดียว
                        conf_thres = cfg.get("confidence")
                        predict_results = model_B.predict(
                            source=person_crop, 
                            verbose=False,
                            conf=conf_thres,
                            classes=None if not class_selected else class_selected
                        )[0]
                        class_ids = predict_results.boxes.cls.cpu().numpy()
                        
                        people_detections.append({
                            "predict_id": str(uuid.uuid4()),
                            "filename": filename,
                            "timestamp": timestamp,
                            "width": width,
                            "height": height,
                            "track_id": int(tid),
                            "x_person": x1,
                            "y_person": y1,
                            "w_person": x2 - x1,
                            "h_person": y2 - y1
                        })

                        
                        for idx, pbox in enumerate(predict_results.boxes):
                            cls_b = int(pbox.cls[0])
                            conf_b = float(pbox.conf[0])
                            x1_clothing, y1_clothing, x2_clothing, y2_clothing = map(int, pbox.xyxy[0])
                            crop_clothing = person_crop[y1_clothing:y2_clothing, x1_clothing:x2_clothing]
                            mean_color_bgr = get_color_dominant(crop_clothing)

                            detections.append({
                                "predict_id": str(uuid.uuid4()),
                                "filename": filename,
                                "width": width,
                                "height": height,
                                "timestamp": timestamp,
                                "class": int(cls_b),
                                "class_name": CLASS_NAMES_B[cls_b],
                                "confidence": round(conf_b, 2), 
                                "x_person": x1,
                                "y_person": y1,
                                "w_person": x2 - x1,
                                "h_person": y2 - y1,
                                "x_clothing": x1_clothing,
                                "y_clothing": y1_clothing,
                                "w_clothing": x2_clothing - x1_clothing,
                                "h_clothing": y2_clothing - y1_clothing,
                                "track_id": int(tid),
                                "mean_color_bgr": mean_color_bgr
                            })
                        data_cleaned = clean_data(detections)
                        clean_detections_clothing.extend(data_cleaned)
        frame_index += 1

    cap.release()

    # Save results
    fieldnames_people = [
            "predict_id","filename","timestamp","track_id","width","height","x_person","y_person","w_person","h_person"
        ]
    save_result(resultpeople_detection_csv,fieldnames_people,people_detections)
    
    fieldnames_clothing = [
            "predict_id", "filename", "width", "height", "timestamp",
            "class", "class_name", "confidence",
            "x_person", "y_person", "w_person", "h_person",
            "x_clothing", "y_clothing", "w_clothing", "h_clothing",
            "mean_color_bgr", "track_id"
        ]
  
    save_result(output_csv,fieldnames_clothing,clean_detections_clothing)
    if class_selected==None:
        config.update({"CSV_PEOPLE_DETECTION": resultpeople_detection_csv})
        config.update({"CSV_R_PATH": output_csv})
    else:
        config.update({"CSV_CUSTOM_RESULT_PERSON": resultpeople_detection_csv})
        config.update({"CSV_CUSTOM_RESULT_CLOTHING": output_csv})
    
    save_config(config)
    return clean_detections_clothing
# ========================================
# Save result
# ========================================

def save_result(output_csv,fieldnames,result):
    file_exists = os.path.exists(output_csv)    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in result:
            writer.writerow(row)

# def detect_objects(video_path, model_A, model_B, model_C, output_csv, cfg):
    
#     cap = cv2.VideoCapture(video_path)

#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_index = 0
    
#     clean_detections = []
#     seen_track_ids = set()
#     frame_interval = int(fps * cfg["frequency"])  # ตรวจจับทุก cfg["frequency"] วินาที

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break

#         if frame_index % frame_interval == 0:    
#             # Track ด้วย model_A (เฉพาะ class person=0)
#             track_results = model_A.track(
#                 source=frame,
#                 persist=True,
#                 classes=[0],
#                 verbose=False,
#                 tracker='botsort.yaml'
#             )[0]
            
#             if track_results.boxes.id is not None:
#                 for box, tid in zip(track_results.boxes.xyxy, track_results.boxes.id):
#                     detections = []
#                     x1, y1, x2, y2 = map(int, box.tolist())
#                     person_crop = frame[y1:y2, x1:x2]
                    
#                     # Predict เสื้อผ้าเฉพาะเมื่อมี track_id ใหม่
#                     if tid not in seen_track_ids:
#                         # seen_track_ids.add(tid) # ถ้าต้องการให้ตรวจครั้งเดียว
#                         conf_thres = cfg.get("confidence")
#                         predict_results = model_B.predict(
#                             source=person_crop, 
#                             verbose=False,
#                             conf=conf_thres
#                         )[0]
          
                        
#                         for idx, pbox in enumerate(predict_results.boxes):
#                             cls_b = int(pbox.cls[0])
#                             conf_b = float(pbox.conf[0])

#                             x1_clothing, y1_clothing, x2_clothing, y2_clothing = map(int, pbox.xyxy[0])
                            

#                             detections.append({
#                                 "filename": os.path.basename(video_path),
#                                 "width": width,
#                                 "height": height,
#                                 "timestamp": round(frame_index / fps, 2),
#                                 "class": cls_b,
#                                 "class_name": CLASS_NAMES_B[cls_b],
#                                 "confidence": round(conf_b, 2), 
#                                 "x_person": x1,
#                                 "y_person": y1,
#                                 "w_person": x2 - x1,
#                                 "h_person": y2 - y1,
#                                 "x_clothing": x1_clothing,
#                                 "y_clothing": y1_clothing,
#                                 "w_clothing": x2_clothing - x1_clothing,
#                                 "h_clothing": y2_clothing - y1_clothing,
#                                 "track_id": int(tid),
#                                 "mean_color_bgr": None
#                             })
#                         data_cleaned = clean_data(detections)
#                         for data in data_cleaned:
#                             class_crop = person_crop[data["y_clothing"]:data["y_clothing"]+data["h_clothing"], data["x_clothing"]:data["x_clothing"]+data["w_clothing"]]
#                             color = get_color(model_C, class_crop,data["w_clothing"],data["h_clothing"])
#                             data["mean_color_bgr"] = color
#                             if color is not None:
#                                 clean_detections.append(data)
#         frame_index += 1

#     cap.release()

#     # Save results
#     file_exists = os.path.exists(output_csv)
#     os.makedirs(os.path.dirname(output_csv), exist_ok=True)
#     with open(output_csv, 'a', newline='') as csvfile:
#         fieldnames = [
#             "filename", "width", "height", "timestamp",
#             "class", "class_name", "confidence",
#             "x_person", "y_person", "w_person", "h_person",
#             "x_clothing", "y_clothing", "w_clothing", "h_clothing",
#             "mean_color_bgr", "track_id"
#         ]
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         if not file_exists:
#             writer.writeheader()
#         for row in clean_detections:
#             writer.writerow(row)

#     config.update({"CSV_R_PATH": output_csv})
#     save_config(config)

# ===============================
# MAIN PROCESS FUNCTION
# ===============================
def process_videos(detect_all=True, custom_config=None):
    cfg = load_config_(custom_config)
    model_A = YOLO(cfg["TRACKING_AI"])
    model_B = YOLO(cfg["AI_MODEL_PATH"])
    print("start detect_objects")
    print("model A: ",cfg["TRACKING_AI"])
    print("model B: ",cfg["AI_MODEL_PATH"])
    video_files = get_video_files([cfg["VIDEO_PATH"]])
    dir = cfg["RESULTS_PREDICT_DIR"]
    processed = set()
    scanned_file = 0


    output_csv = get_result_csv(dir,detect_all,"clothing_detection")
    resultpeople_detection_csv = get_result_csv(dir,detect_all,"people_detection")

    if not detect_all:
        processed = load_processed_files()

    for video in video_files:
        start_track = time.perf_counter()
        filename = os.path.basename(video)
        if not detect_all and filename in processed:
            print(f"[✔] Skip processed: {filename}")
            continue
        
            
        print(f"[▶] Processing: {filename}")
        detect_objects(video,model_A, model_B,  output_csv,cfg,resultpeople_detection_csv)
        mark_file_as_processed(filename)
        write_log(filename)
        scanned_file += 1
        track_time = time.perf_counter() - start_track
        print(track_time)
    return scanned_file
    
# ===============================
# INTERRUPT SAFE START
# ===============================
def start_process(detect_all=True, custom_config=None):

    print("[▶] Starting process...")

    scanned_file = process_videos(detect_all, custom_config)
    return {"scanned_file": scanned_file}
    
    return_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=process_videos, args=(detect_all, custom_config, return_queue))
    process.start()

    def handle_interrupt(sig, frame):
        print("\n[!] Interrupt received. Terminating process...")
        process.terminate()
        process.join()
        exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    process.join()
    result = return_queue.get()
    return {"scanned_file": result}
# ===============================
# detecrt-now
# ===============================


if __name__ == "__main__":
    
    # detect_all=True = ตรวจทุกไฟล์
    # detect_all=False = ตรวจเฉพาะไฟล์ที่ยังไม่ตรวจ
    # start_process(detect_all=True)
    # get_video_files(config.get("VIDEO_PATH"))
    # image = cv2.imread(r"E:\picture\white.png")
    # width = image.shape[1]
    # height = image.shape[0]
    # get_color(model_color, image,width,height)
    start_process(detect_all=True)
