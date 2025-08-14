import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO
import os
import time

def timer(func):
    """Decorator สำหรับจับเวลาการทำงานของฟังก์ชัน"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f">>> [{func.__name__}] ใช้เวลา: {end - start:.4f} วินาที")
        return result
    return wrapper

@timer
def load_model(model_path):
    return YOLO(model_path)

@timer
def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"ไม่พบภาพ: {image_path}")
    return image

@timer
def run_prediction(model, image_path):
    return model(image_path)

@timer
def crop_and_save(image, results, image_path, save_dir="crops"):
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    crops = []

    for i, result in enumerate(results):
        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            crop = image[y1:y2, x1:x2]

            crop_name = f"{base_name}_cls{cls}_conf{conf:.2f}_{j}.jpg"
            crop_path = os.path.join(save_dir, crop_name)
            cv2.imwrite(crop_path, crop)
            print(f"บันทึก: {crop_path}")
            crops.append((crop, cls, conf, crop_path))
    return crops

@timer
def get_dominant_color(image):
    pixels = image.reshape((-1, 3))
    pixels = pixels[np.all(pixels > 30, axis=1)]  # ตัดฉากหลังดำ
    if len(pixels) == 0:
        return (0, 0, 0)  # fallback
    counts = Counter([tuple(p) for p in pixels])
    dominant_color = counts.most_common(1)[0][0]
    return dominant_color  # (B, G, R)

@timer
def predict_and_crop(image_path, model_path, save_dir="crops"):
    model = load_model(model_path)
    image = read_image(image_path)
    results = run_prediction(model, image_path)
    crops = crop_and_save(image, results, image_path, save_dir)

    # Optional: ตรวจสี dominant
    for crop, cls, conf, path in crops:

        color = get_dominant_color(crop)
        print(f"→ class {cls}, conf {conf:.2f}, สีเด่น: {color}")

    return crops

# ตัวอย่างการเรียกใช้
if __name__ == "__main__":
    color = predict_and_crop("000376.jpg", r"E:\python\model\fashion13cls.pt")

