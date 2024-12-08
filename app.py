from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import pandas as pd
import csv
import re
import base64

# Flask app initialization
app = Flask(__name__)

# Load YOLOv8 model
yolo_model = YOLO("best.pt").to('cpu')

# Load TrOCR model and processor
trocr_model = VisionEncoderDecoderModel.from_pretrained("models/rayyaa/finetune-trocr")
trocr_processor = TrOCRProcessor.from_pretrained("models/rayyaa/finetune-trocr")


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
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arialbd.ttf", size=25)

    # YOLO Plate Detection
    results = yolo_model(image)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes

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

        # Draw bounding box and plate number on the image
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_min, y_min - 27), plate_number, fill="red", font=font)

    # Convert annotated image to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return JSON response with plates information and annotated image
    response = {
        'plates': plates_info,
        'annotated_image': img_str  # Base64-encoded image
    }
    return jsonify(response)

@app.route('/')
def hello_world():
    return 'Hello, Flask World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
