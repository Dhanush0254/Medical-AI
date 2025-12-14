import re
import logging
import cv2
import numpy as np
import pytesseract
import difflib
import json
import pandas as pd
import pdfplumber
import os
from pathlib import Path
from PIL import Image

# --- SAFETY ---
try:
    from pdf2image import convert_from_path
    PDF_IMAGE_SUPPORT = True
except ImportError:
    PDF_IMAGE_SUPPORT = False

logging.basicConfig(level=logging.INFO)

# --- 1. SMART TESSERACT SETUP ---
if os.name == 'nt': 
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else: 
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# --- 2. MEGA DICTIONARY ---
SEARCH_MAP = {
    'glucose': ['glucose', 'glu', 'sugar', 'fbs', 'bsl', 'fasting blood sugar', 'blood glucose', 'rbs', 'ppbs'],
    'hba1c': ['hba1c', 'a1c', 'glycated', 'haemoglobin a1c', 'hb a1c'],
    'hemoglobin': ['hemoglobin', 'hgb', 'hb', 'haemoglobin'],
    'cholesterol': ['cholesterol', 'chol', 'total cholesterol', 't.chol', 'chorestrol', 'chorestall'],
    'ldl': ['ldl', 'bad cholesterol', 'low density', 'ldl-c'],
    'hdl': ['hdl', 'good cholesterol', 'high density', 'hdl-c'],
    'triglycerides': ['triglycerides', 'trigs', 'tg', 'tgl'],
    'red_blood_cells': ['red_blood_cells', 'rbc', 'erythrocytes', 'red blood', 'total rbc'],
    'age': ['age', 'years', 'y/o', 'yrs', 'patient age']
}

IGNORE_KEYWORDS = ['ref', 'range', 'limit', 'min', 'max', 'interval', 'method', 'date', 'time', 'units']

VALID_RANGES = {
    'glucose': (20, 2000), 'hba1c': (2, 25), 'hemoglobin': (2, 30),
    'cholesterol': (50, 1000), 'ldl': (10, 800), 'hdl': (5, 300),
    'triglycerides': (10, 3000), 'red_blood_cells': (0.5, 15), 'age': (1, 120)
}

def clean_value(val):
    try:
        val_str = str(val).lower().strip()
        val_str = re.sub(r'(high|low|mg/dl|g/dl|mmol/l)', '', val_str)
        clean = re.sub(r'[^\d\.]', '', val_str)
        if clean.count('.') > 1: clean = clean.replace('.', '', clean.count('.') - 1)
        return float(clean)
    except: return None

def map_key_to_standard(text):
    text_lower = str(text).lower().replace('_', ' ').strip()
    if any(bad in text_lower for bad in IGNORE_KEYWORDS): return None

    for std_key, variants in SEARCH_MAP.items():
        if any(v in text_lower for v in variants): return std_key
        if difflib.get_close_matches(text_lower, variants, cutoff=0.80): return std_key
    return None

# --- 3. DEEP STRUCTURE SCAN ---
def scan_complex_structure(data):
    results = {}
    def recursive_scan(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                std_key = map_key_to_standard(k)
                if std_key:
                    val = clean_value(v)
                    if val and VALID_RANGES.get(std_key)[0] <= val <= VALID_RANGES.get(std_key)[1]:
                        if std_key not in results: results[std_key] = val
                
                if isinstance(v, str):
                    key_from_val = map_key_to_standard(v)
                    if key_from_val:
                         for sib_k, sib_v in obj.items():
                            if sib_k == k: continue
                            val = clean_value(sib_v)
                            if val and VALID_RANGES.get(key_from_val)[0] <= val <= VALID_RANGES.get(key_from_val)[1]:
                                results[key_from_val] = val
                recursive_scan(v)
        elif isinstance(obj, list):
            for item in obj: recursive_scan(item)
    recursive_scan(data)
    return results

# --- 4. MATRIX SCAN ---
def scan_csv_matrix(df):
    data = {}
    df_str = df.astype(str)
    
    for index, row in df_str.iterrows():
        for col_name, cell_val in row.items():
            std_key = map_key_to_standard(cell_val)
            if std_key:
                for other_col in df.columns:
                    if other_col == col_name: continue
                    if any(bad in str(other_col).lower() for bad in IGNORE_KEYWORDS): continue
                    
                    val = clean_value(row[other_col])
                    if val and VALID_RANGES.get(std_key)[0] <= val <= VALID_RANGES.get(std_key)[1]:
                        data[std_key] = val
                        break 
    return data

# --- 5. OPTIMIZED IMAGE PREP (The Speed Fix) ---
def preprocess_image(image):
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
            if len(image.shape) == 3: image = image[:, :, ::-1].copy()
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # SPEED FIX 1: Don't upscale if image is already readable (>1000px)
        # SPEED FIX 2: Downscale huge phone photos (>2500px) to save CPU
        h, w = gray.shape
        if w < 1000:
            gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        elif w > 2500:
            scale = 2500 / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Standard Thresholding (Faster than adaptive)
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    except: return image

# --- 6. PDF SCANNER ---
def extract_pdf_content(path):
    full_text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        full_text += " ".join([str(x) for x in row if x]) + "\n"
                
                text = page.extract_text()
                
                has_keywords = any(k in (text or "").lower() for k in SEARCH_MAP.keys())
                
                if (not text or len(text) < 20 or not has_keywords) and PDF_IMAGE_SUPPORT:
                    # Limit DPI to 150 for speed (Standard is 200+)
                    images = convert_from_path(str(path), first_page=i+1, last_page=i+1, dpi=150)
                    for img in images:
                        processed = preprocess_image(img)
                        full_text += pytesseract.image_to_string(processed, config='--psm 6') + "\n"
                else:
                    full_text += (text or "") + "\n"
    except: pass
    return full_text

# --- 7. OCR PARSER ---
def parse_text_content(text):
    data = {}
    lines = text.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        clean_line = re.sub(r'\d+\s*-\s*\d+', '', line_lower)
        clean_line = re.sub(r'[<>]\s*\d+', '', clean_line)

        for std_key, variants in SEARCH_MAP.items():
            if any(v in clean_line for v in variants):
                matches = re.findall(r'\b\d+(?:\.\d+)?\b', clean_line)
                valid = []
                for m in matches:
                    try:
                        val = float(m)
                        if VALID_RANGES.get(std_key)[0] <= val <= VALID_RANGES.get(std_key)[1]:
                            valid.append(val)
                    except: pass
                
                if valid:
                    if std_key not in data: data[std_key] = valid[0]
    return data

# --- 8. MAIN ---
def extract_data(file_path):
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix == '.csv':
        try: return scan_csv_matrix(pd.read_csv(path))
        except Exception as e: return {"error": f"CSV Error: {e}"}

    if suffix == '.json':
        try:
            with open(path, 'r') as f: return scan_complex_structure(json.load(f))
        except Exception as e: return {"error": f"JSON Error: {e}"}

    try:
        if suffix == '.pdf': text = extract_pdf_content(path)
        else:
            img = cv2.imread(str(path))
            processed = preprocess_image(img)
            text = pytesseract.image_to_string(processed, config='--psm 6')
        return parse_text_content(text)
    except Exception as e: return {"error": f"Extraction Error: {e}"}
