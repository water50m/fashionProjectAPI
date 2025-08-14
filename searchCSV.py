
import pandas as pd
from typing import List, Optional
from config import load_config
import os
import numpy as np


config = load_config()
# โหลด CSV ไว้ล่วงหน้า (หรือโหลดจาก parameter ได้)
DF = pd.read_csv(config.get("CSV_R_PATH"))  # แก้ชื่อไฟล์ตามจริง
last_modified_DF = os.path.getmtime(config.get("CSV_R_PATH"))
if DF is not None:
    print("Summary result loaded successfully")
else:
    print("Summary result not found")

def filter_by_filename(df: pd.DataFrame, filenames: List[str]) -> pd.DataFrame:
    return df[df["filename"].isin(filenames)]


# filter by class non-collab in self
def filter_single_class(df: pd.DataFrame, classes:  List[str]) -> pd.DataFrame:

    """
    กรองเฉพาะแถวที่ class_name อยู่ใน classes
    """
    return df[df["class_name"].isin(classes)]

# filter by class with collab 
def filter_collab_class(df: pd.DataFrame, classes:  List[str]) -> pd.DataFrame:
    """
    กรองเฉพาะ timestamp ที่มีครบทุก class ที่ระบุใน classes
    """
    matched_frames = []
    grouped = df.groupby(["filename", "timestamp"])
    for (filename, ts), group in grouped:
        class_names = set(group["class_name"])
        if all(cls in class_names for cls in classes):
            matched_frames.append(group[group["class_name"].isin(classes)])
    
    return pd.concat(matched_frames) if matched_frames else pd.DataFrame(columns=df.columns)


def filter_by_date(df: pd.DataFrame, date: Optional[str]) -> pd.DataFrame:
    if "date" not in df.columns or date is None:
        return df
    return df[df["date"] == date]

def prepare_and_find_similar_colors(df: pd.DataFrame, input_colors_hex, threshold):
    # แปลง hex เป็น rgb
    def hex_to_rgb(hex_color: str):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # แปลง bgr string เป็น rgb tuple
    def bgr_to_rgb(bgr_str):
        nums = bgr_str.strip('[]').split()
        bgr = [float(n) for n in nums]  
        return (int(bgr[2]), int(bgr[1]), int(bgr[0]))

    # คำนวณระยะห่างสี
    def color_distance(rgb1, rgb2):
        return sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5

    # แปลง input colors hex เป็น rgb
    input_colors_rgb = [hex_to_rgb(c) for c in input_colors_hex]


    df.loc[:, 'mean_color_rgb'] = df['mean_color_bgr'].apply(bgr_to_rgb)

    # หาค่าสีที่ใกล้เคียงกัน
    result_rows = []
    for rgb in input_colors_rgb:
        for  _,row in df.iterrows():
            dist = color_distance(rgb, row['mean_color_rgb'])
            if dist <= threshold:
                result_rows.append(row)
    if result_rows:
        return pd.DataFrame(result_rows)
    else:
        return pd.DataFrame(columns=df.columns)


def search_csv(data) -> pd.DataFrame:
    global DF
    global last_modified_DF
    # ตรวจสอบว่าไฟล์ csv ถูกแก้ไขหรือไม่
    check_modified = os.path.getmtime(config.get("CSV_R_PATH"))
    if check_modified != last_modified_DF:
        DF = pd.read_csv(config.get("CSV_R_PATH"))  # แก้ชื่อไฟล์ตามจริง
        last_modified_DF = check_modified
    try:
        df = DF.copy()
        filtered=False
        # หาจากชื่อ file
        if data.filename:
            df = filter_by_filename(df, data.filename)
            filtered=True  
        
        if data.class_color:
            cls=[]
            cls_clr=[]
            for item in data.class_color:
                cls.append(item["class"])              
                filteredList = filter_single_class(df, [item["class"]])
                if item['colors']:
                    filteredList = prepare_and_find_similar_colors(filteredList,item['colors'], threshold=100)
                cls_clr.append(filteredList)
            df = pd.concat(cls_clr)
            if data.class_collab:
                df = filter_collab_class(df, cls)
            filtered=True

    except Exception as e:
        print("เกิดข้อผิดพลาด:", e)
        return []
    
    if df.empty:
        print("ไม่พบข้อมูลที่ตรงกับเงื่อนไข")
    else:
        print("ผลลัพธ์ที่ค้นพบ:")
        # print(df[:10])
    if filtered:
        return df.to_dict(orient="records")
    else:
        return []