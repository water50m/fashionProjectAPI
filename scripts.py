# script.py
from config import load_config
import os
import pandas as pd
import cv2
from collections import defaultdict


config = load_config()

def scan_folder(folder_path,target_extensions):
    try:

        file_list = [
            entry.name
            for entry in os.scandir(folder_path)
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in target_extensions
        ]

        # print(f"เจอทั้งหมด {len(file_list)} ไฟล์")
        return file_list 
    except Exception as e: 
        print(f"Error: {e}")

def scan_ai_model():
    folder_path = config.get("AI_MODEL_PATH", "")
    target_extensions = {".pt"}  # ใส่ชนิดไฟล์ที่ต้องการ
    return scan_folder(folder_path,target_extensions)

       

def scan_video_floder():
    folder_path = config.get("VIDEO_PATH", "")
    target_extensions = {".mp4", ".avi", ".mov"}  # ใส่ชนิดไฟล์ที่ต้องการ
    return scan_folder(folder_path,target_extensions)

def check_video_unscanned():
    predicted_video = config.get("PROCESSED_LOG", "")
    with open(predicted_video, 'r') as f:
        predicted_video_name = f.read()
        current_video = scan_video_floder()
        yet_predicted = list(set(current_video) - set(predicted_video_name.split("\n")))
        return yet_predicted

def scan_all():
    current_path=config.get("VIDEO_PATH", "")
    model_ai=scan_ai_model()
    video_floder=scan_video_floder()
    video_unscanned=check_video_unscanned()
    predicted_file_list=list(set(video_floder) - set(video_unscanned))
    # default config
    confidence_threshole=config.get("MODEL_CONFIG")["confidence_threshold"]
    frequency=config.get("MODEL_CONFIG")["frequency"]
    detectingAll=config.get("MODEL_CONFIG")["detectingAll"]
    return {"current_path":current_path,
            "available_models":model_ai,
            "total_files":len(video_floder),
            "unpredicted_files":len(video_unscanned),
            "predicted_files":len(predicted_file_list),
            "unpredicted_file_list":video_unscanned,
            "confidence_threshole":confidence_threshole,
            "frequency":frequency,
            "detectingAll":detectingAll}

# ==============================================================================================
# Function: Load detection data
# ==============================================================================================


CLASS_NAMES = [
    'short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear',
    'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress',
    'long sleeve dress', 'vest dress', 'sling dress'
]

def load_detection_data():
    """Load and parse the CSV detection data"""

    try:

        df = pd.read_csv(config.get("CSV_R_PATH", ""))
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def extract_frame_and_crop(video_path: str, timestamp: float, x: int, y: int, w: int, h: int, 
                          output_path: str) -> bool:
    """Extract frame from video at timestamp and crop the specified region"""
    try:
        cap = cv2.VideoCapture(video_path)
        

        # Set video position to timestamp
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        if ret:
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Ensure crop coordinates are within frame bounds
            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            w = max(1, min(w, frame_width - x))
            h = max(1, min(h, frame_height - y))
            
            # Crop the image
            cropped = frame[y:y+h, x:x+w]
            
            # Save the cropped image
            cv2.imwrite(output_path, cropped)
            cap.release()
            return True
        
        cap.release()
        return False
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return False

def process_track_data(df: pd.DataFrame, video_name: str):
    """Process detection data for a specific video and generate track analysis"""
    
    # Filter data for the specific video
    video_df = df[df['filename'] == video_name].copy()
    
    if video_df.empty:
        return {
            "video_name": video_name,
            "tracks": [],
            "all_classes": []
        }
    
    tracks_data = []
    all_classes_set = set()
    
    # Group by track_id
    for track_id, track_group in video_df.groupby('track_id'):
        # Calculate total confidence for each class
        class_confidences = defaultdict(float)
        class_data = defaultdict(list)
        
        for _, row in track_group.iterrows():
            class_name = row['class_name']
            confidence = row['confidence']
            # print("row",row)
            class_confidences[class_name] += confidence
            try:
                class_data[class_name].append({
                'timestamp': row['timestamp'],
                'confidence': confidence,
                'x_person': row['x_person'],
                'y_person': row['y_person'],
                'w_person': row['w_person'],
                'h_person': row['h_person']
            })
            except Exception as e:
                print(f"Error processing track data: {e}")
            all_classes_set.add(class_name)

        # Get top 2 classes by total confidence
        sorted_classes = sorted(class_confidences.items(), key=lambda x: x[1], reverse=True)
 
        top_classes = [cls[0] for cls in sorted_classes[:2]]

        if not top_classes:
            continue
            
        # Find best timestamp for the top class
        best_class = top_classes[0]
        best_detection = max(class_data[best_class], key=lambda x: x['confidence'])
        best_timestamp = best_detection['timestamp']
        

        # Generate example image
        video_path = os.path.join(config.get("VIDEO_PATH", ""), video_name)
        image_filename = f"{os.path.splitext(video_name)[0]}-{track_id}-{best_timestamp:.1f}-{best_class.replace(' ', '_')}.jpg"
        image_path = os.path.join(config.get("EXAMPLE_PICS_PATH", ""), image_filename)

        
        # Extract and crop frame
        success = extract_frame_and_crop(
            video_path,
            best_timestamp,
            int(best_detection['x_person']),
            int(best_detection['y_person']),
            int(best_detection['w_person']),
            int(best_detection['h_person']),
            image_path
        )
        
        # Calculate time range for this track
        timestamps = track_group['timestamp'].values
        time_range = {
            'start': float(timestamps.min()),
            'end': float(timestamps.max())
        }
        
        tracks_data.append({
            'track_id': int(track_id),
            'classes': top_classes,
            'total_confidences': {cls: float(conf) for cls, conf in class_confidences.items()},
            'best_timestamp': float(best_timestamp),
            'best_class': best_class,
            'time_range': time_range,
            'example_image_url': f"{config.get('FAST_API', "http://localhost:8000")}/example_pics/{image_filename}" if success else None,
            'bbox': {
                'x': int(best_detection['x_person']),
                'y': int(best_detection['y_person']),
                'w': int(best_detection['w_person']),
                'h': int(best_detection['h_person'])
            }
        })
    print("for track_id, track_group in video_df.groupby('track_id'):")
    return {

        'video_name': video_name,
        'tracks': tracks_data,
        'all_classes': list(all_classes_set)
    }


def get_data_from_csv(video_name):
    df = pd.read_csv(config.get("CSV_R_PATH", ""))
    df = df[df['filename'] == video_name]

    return df

def get_data_people_from_csv(video_name):
    df = pd.read_csv(config.get("CSV_PEOPLE_DETECTION", ""))
    df = df[df['filename'] == video_name]

    return df


def get_all_data():
    df = pd.read_csv(config.get("CSV_R_PATH", ""))

    return df


def get_all_data_people():
    df = pd.read_csv(config.get("CSV_PEOPLE_DETECTION", ""))

    return df







from ultralytics import YOLO


def check_model():
    model = YOLO("E:\\ALL_CODE\\nextjs\\fashion_project_1\\scripts\\model\\segmentModel.pt")
    print("Task type:", model.task)

def test_model():
    frame = cv2.imread(r"E:\ALL_CODE\nextjs\fashion_project_1\public\picture\uploaded-pic\1754223336856_rnx715_trousers.jpg")

    # 1. Load YOLO segmentation model
    model =  YOLO("E:\\ALL_CODE\\nextjs\\fashion_project_1\\scripts\\model\\segmentModel.pt")  # or your custom trained segmentation model

    # 2. Run segmentation prediction
    results = model.predict(
        source=frame,   # image, video, folder, or webcam index
        save=True,            # save output
        show=True,            # display in a window
        conf=0.5              # confidence threshold
    )
    print(results)
    print("----------------------------------------------------")
    print("----------------------------------------------------")
    # 3. Process results
    for r in results:
        boxes = r.boxes.xyxy       # bounding box [x1, y1, x2, y2]
        classes = r.boxes.cls      # class indices
        masks = r.masks.data       # segmentation masks (tensor) [n, h, w]

        h, w = frame.shape[:2]

        # แปลง mask เป็น numpy และ resize ให้ตรงกับ frame
        mask = masks[0].cpu().numpy()
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # แปลงเป็น boolean
        mask_bool = mask_resized.astype(bool)

        # ดึงพิกเซล
        pixels_in_mask = frame[mask_bool]
        mean_color_bgr = pixels_in_mask.mean(axis=0)
        mean_color_rgb = mean_color_bgr[::-1]

        print("ค่าเฉลี่ยสี (RGB):", mean_color_rgb)

        print("Boxes:", boxes)
        print("Classes:", classes)
        print("Masks shape:", masks.shape)  # e.g., (n, 640, 640)

        # Convert mask to numpy for OpenCV
        for mask in masks:
            mask_np = mask.cpu().numpy().astype("uint8") * 255
            cv2.imshow("Mask", mask_np)
            cv2.waitKey(0)

if __name__ == "__main__":
    get_all_data_people() 