from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import pandas as pd
import csv
import re
import torch
import os
import io

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Load YOLOv8 model
yolo_model = YOLO("best.pt")

# Load TrOCR model and processor
trocr_model = VisionEncoderDecoderModel.from_pretrained("rayyaa/finetune-trocr")
trocr_processor = TrOCRProcessor.from_pretrained("rayyaa/finetune-trocr")

def perform_ocr(cropped_image):
    # Preproses gambar untuk TrOCR
    pixel_values = trocr_processor(images=cropped_image, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

# Load plate region mapping
plat_nomor_wilayah = {}
file_path = 'plates_region.csv'

with open(file_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        kode = row['code'].strip().upper()  # Kolom 'code'
        area = row['area'].strip()         # Kolom 'area'
        plat_nomor_wilayah[kode] = area

def get_plate_region(plat_text):
    """Find region based on plate code."""
    match = re.match(r'^[A-Z]{1,2}', plat_text.strip().upper())
    if match:
        kode_awal = match.group(0)
        return plat_nomor_wilayah.get(kode_awal, "Wilayah tidak ditemukan")

@app.route('/predict', methods=['POST'])
def predict():
    # Check if file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    # Get the uploaded image
    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')

    # YOLO Plate Detection
    results = yolo_model(image)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes

    # List untuk menyimpan hasil
    predictions = []
    
    if len(detections) == 0:
        return jsonify({'error': 'No plate detected'}), 404

    # Process all detected plates
    plates_info = []
    for det in detections:
        x_min, y_min, x_max, y_max = det
        plate_image = image.crop((x_min, y_min, x_max, y_max))

        # TrOCR OCR Prediction
        plate_text = trocr_processor(plate_image, return_tensors="pt").pixel_values
        decoded_text = trocr_model.generate(plate_text)
        plate_number = trocr_processor.batch_decode(decoded_text, skip_special_tokens=True)[0]

        # Find region
        region = get_plate_region(plate_number)

        # Append plate info
        plates_info.append({'plate_number': plate_number, 'region': region})

    return jsonify({'plates': plates_info})

@app.route('/')
def hello_world():
    return 'Hello, Flask World!'

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)

