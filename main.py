# main.py


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from searchCSV import search_csv,prepare_and_find_similar_colors
from prediction_img import YOLOPredictor
from fastapi import File, UploadFile
import tempfile
from scripts import scan_all, load_detection_data, process_track_data, get_data_from_csv, get_all_data, get_data_people_from_csv, get_data_csv_with_path
from multiprocessing import Event
from detection_tracking_save import start_process,detect_objects,get_result_csv,load_config_
from config import load_config
from fastapi.staticfiles import StaticFiles
import os
from config import load_config, save_config, CONFIG_FILE
import pandas as pd
from ultralytics import YOLO
import traceback
import numpy as np
import sys

config = load_config()
process = None
pause_event = Event()
stop_event = Event()

app = FastAPI()

# ตั้งค่าการอนุญาต CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # หรือใช้ ["*"] ชั่วคราวเพื่อทดสอบ
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Ensure example pics directory exists
os.makedirs(config.get("EXAMPLE_PICS_PATH", ""), exist_ok=True)

# Mount static files
app.mount("/example_pics", StaticFiles(directory=config.get("EXAMPLE_PICS_PATH", "")), name="example_pics")

class PythonInput(BaseModel):
    filename: list[str]
    class_color: list[dict]
    class_collab: bool
    date: Optional[str] = None
    color_collab: bool

class configItem(BaseModel):
    CONFIG_FILE_PATH: str
    CSV_R_PATH: str
    VIDEO_PATH: str
    AI_MODEL_PATH: str
    AI_MODEL_NAME: str
    PROCESSED_LOG: str
    RUNNING_PATH: str
    DETECTION_RESULT_FILE: str

class ScannerData(BaseModel):
    auto_config: bool
    video_path: str
    use_system_ai: bool
    system_model: str
    custom_ai_path: str
    confidence: float
    frequency: float
    detectingAll: bool



@app.post("/run-python1")
async def run_python(data: PythonInput):
    from subprocess import Popen, PIPE

    process = Popen(['python', 'scripts.py'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, error = process.communicate(input=data.json().encode('utf-8'))

    if process.returncode == 0:
        return { "output": output.decode().strip() }
    else:
        return { "error": error.decode().strip() }



@app.post("/finding")
async def finder(data:PythonInput):
    try:
        # data เป็น PythonInput instance อยู่แล้ว
        # ถ้าต้องการ dict ใช้ data.dict()
        result = search_csv(data)
        return result
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------configuration--part------------------------------------




@app.get("/api/config/{key}")
def get_config_item(key: str):
    
    return {"key": key, "value": config.get(key)}

@app.get("/api/config")
def get_all_config_items():
    config["CONFIG_FILE_PATH"] = CONFIG_FILE
    return config

@app.post("/api/config-update")
def update_config_item(item: configItem):
    item_dict = item.dict(exclude_unset=True)  # เฉพาะ field ที่ถูกส่งมา
    config.update(item_dict)  # update เฉพาะ key ที่ส่งมา
    save_config(config)
    return {"message": "updated", "updated_fields": item_dict}





# ---------------------------predict picture part--------------------------------

class predictPicture(BaseModel):
    image: str
    filename: str
    class_color: list[dict]
    class_collab: bool
    date: Optional[str] = None
    color_collab: bool
      

@app.post("/predict-picture")
def predict_picture(data:predictPicture):
    try:
        print('predict-picture API success')
        print(data)
    except Exception as e:
        return {"error": str(e)}


# Initialize predictor
predictor = YOLOPredictor()

@app.post("/predict-pic")
async def predict_picture(file: UploadFile = File(...)):
    """
    Predict objects in uploaded image
    
    Args:
        file: Uploaded image file (formData)
    
    Returns:
        List of predictions with format:
        [{'classname': 'name_of_class', 
          'colors': ['color1', 'color2'], 
          'Keypoints_label': [yolo_keypoint_format], 
          'B_Box_label': [yolo_bbox_format]}]
    """
     
    # Validate file type
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        try:
            # Save uploaded file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Run prediction
            predictions = predictor.predict_image(temp_file.name)
            
            return JSONResponse(content=predictions)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": predictor.model is not None}

@app.get("/model-info")
async def model_info():
    """Get model information"""
    try:
        return {
            "model_path": os.path.join(
                predictor.config.get("AI_MODEL_PATH", ""),
                predictor.config.get("AI_MODEL_NAME", "")
            ),
            "classes": predictor.model.names,
            "config": predictor.config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------scanning part--------------------------------
@app.get("/scanner")
def scanner():
    try:
        print('scnaer API success')
        return scan_all()
    except Exception as e:
        return {"error": str(e)}





# @app.post("/scanning")
# def scanning(data:ScannerData):

#     global process
#     if process and process.is_alive():
#         return {"message": "Already running"}
    
#     stop_event.clear()
#     pause_event.clear()
#     process = Process(target=detection(data))
#     process.start()
#     return {"message": "Started"}

# @app.get("/scanning-pause")
# def scanning_pause():
#     pause_event.set()
#     return {"message": "Paused"}

# @app.get("/scanning-resume")
# def scanning_resume():
#     pause_event.clear()
#     return {"message": "Resumed"}

# @app.get("/scanning-stop")
# def scanning_stop():
#     stop_event.set()
#     if process:
#         process.join()
#     return {"message": "Stopped"}


@app.post("/scanning")
def scanning(data: ScannerData):
    try:
        print('scanning API success', data)
        if data.auto_config:
            print("sys config")
            return start_process(detect_all=data.detectingAll)
        else:
            print("custom config")
            return start_process(detect_all=data.detectingAll, custom_config=data)

    except Exception as e:
        # ดึงข้อมูลไฟล์และบรรทัดที่ error
        exc_type, exc_obj, exc_tb = sys.exc_info()
        filename = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        error_message = f"{str(e)} (File: {filename}, Line: {line_number})"
        print(error_message)
        # ใช้ HTTPException เพื่อเปลี่ยน status code
        raise HTTPException(status_code=500, detail=error_message)


# ---------------------------------------Function: advance-finding-------------------------------------------------------

@app.get("/api/videos")
async def get_videos():
    """Get list of available videos"""
    try:
        videos = []
        if os.path.exists(config.get("VIDEO_PATH", "")):
            for file in os.listdir(config.get("VIDEO_PATH", "")):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    videos.append(file)  # Just the filename, not full path
        
        print(f"Found videos: {videos}")  # Debug log
        return sorted(videos)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading video directory: {str(e)}")


@app.get("/api/analyze-video/{video_name}")
async def analyze_video(video_name: str):
    """Analyze detection data for a specific video"""
    try:
        
        # Load detection data
        df = load_detection_data()

        if df is None:
            raise HTTPException(status_code=500, detail="Could not load detection data")
        
        # Check if video exists in data
        if video_name not in df['filename'].values:
            raise HTTPException(status_code=404, detail=f"No detection data found for video: {video_name}")
        
        # Check if video file exists
        video_path = os.path.join(config.get("VIDEO_PATH", ""), video_name)

        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_name}")

        # Process the data
        result = process_track_data(df, video_name)
        
        # Add video duration if possible
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            result['video_duration'] = duration
            cap.release()
        except:
            pass
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing video: {str(e)}")


@app.get("/read-csv/{video_name}")
def read_csv(video_name: str):
    try:
        print(f'Reading CSV for {video_name}')
        data = get_data_from_csv(video_name)

        # Handle nan values by replacing them with None (which becomes null in JSON)
        return data.to_dict(orient="records")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
@app.get("/get-label-tracking-person/{video_name}")
def get_label_tracking_person(video_name: str):
    try:
        data = get_data_people_from_csv(video_name)
        # Handle nan values by replacing them with None (which becomes null in JSON)
        return data.to_dict(orient="records")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/read-all-csv")
def read_csv():
    try:
        data = get_all_data()
        # Handle nan values by replacing them with None (which becomes null in JSON)
        return data.to_dict(orient="records")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")



# ---------------------------custom_detection-----------------------------------------------------------------
class classAndColor(BaseModel):
    classId: int
    colors: list[str]

class CustomDetectionData(BaseModel):
    model: str
    date_range: str
    custom_detection_data: list[classAndColor] 
    videos: list[str]

@app.post("/custom-detection-start")
def custom_detection_start(data:CustomDetectionData):

    try:

        model_A = YOLO(config.get("PERSON_TRACK_PATH", ""))
        model_B = YOLO(config.get("AI_MODEL_PATH", "")+"/"+data.model)
        path = config.get("VIDEO_PATH", "")
        dir = config.get("CSV_CUSTOM_RESULT_PREDICT_DIR", "")
         
        output_csv = get_result_csv(dir,True,"clothing_detection")
        resultpeople_detection_csv = get_result_csv(dir,True,"people_detection")
        class_selected = []
        result = []
        cfg = load_config_()
        for class_ in data.custom_detection_data:
            class_selected.append(class_.classId)

        for video in data.videos:
            video_path = os.path.join(path, video)
            detect = detect_objects(video_path ,model_A ,model_B ,output_csv,cfg,resultpeople_detection_csv,class_selected)
            result.extend(detect)

        if not result:
            return []
            
        df = pd.DataFrame(result)
        filtered_results = []
        
        if data.custom_detection_data:
            for item in data.custom_detection_data:
                print(f'Processing class ID: {item.classId} with colors: {item.colors}')
                filtered = df[df['class'] == item.classId].copy()
                print("filtered",filtered)
                if not filtered.empty:
                    print(f'Found {len(filtered)} matches for class ID {item.classId}')
                    try:
                        filtered = prepare_and_find_similar_colors(filtered, item.colors, 100)
                        filtered_results.append(filtered)
                    except Exception as e:
                        print(f'Error processing class {item.classId}: {str(e)}')
                        continue
        
        if filtered_results:
            print(f'Found {len(filtered_results)} matches for class ID {item.classId}')
            try:
                final_df = pd.concat(filtered_results, ignore_index=True)
                return final_df.to_dict(orient="records")
            except Exception as e:
                print(f'Error concatenating results: {str(e)}')
                return []
        else:
            for col in df.columns:
                if np.issubdtype(df[col].dtype, np.number):
                    print(col, df[col].dtype)
            # If no custom detection data or no matches, return all results
            print("df",df)
            return df.to_dict(orient="records")
    except Exception as e:
        # เก็บ stack trace ทั้งหมดเป็น string
        full_traceback = traceback.format_exc()
        print(full_traceback)  # log เต็ม

        # ดึงข้อมูลไฟล์และบรรทัดที่เกิด error
        exc_type, exc_obj, exc_tb = sys.exc_info()
        filename = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        # รวมข้อความสรุป + stack trace
        error_message = f"Error: {str(e)}\nFile: {filename}, Line: {line_number}\n\nStack Trace:\n{full_traceback}"

        # ส่งกลับใน HTTPException
        raise HTTPException(status_code=500, detail=error_message)

   

if __name__ == "__main__":
    try:
        config = predictor.config
        api_config = config.get("API_CONFIG", {})
        
        print("Starting YOLO Prediction API Server...")
        print(f"Host: {api_config.get('host', '0.0.0.0')}")
        print(f"Port: {api_config.get('port', 8000)}")
        print(f"Model: {config.get('AI_MODEL_PATH', '')}/{config.get('AI_MODEL_NAME', '')}")
        
        uvicorn.run(
            app,
            host=api_config.get("host", "0.0.0.0"),
            port=api_config.get("port", 8000),
            reload=api_config.get("reload", True),
            log_level=api_config.get("log_level", "info")
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        raise

