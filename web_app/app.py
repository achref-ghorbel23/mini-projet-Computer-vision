from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import base64
import numpy as np
from detector import process_image
import time

app = Flask(__name__)

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = None
        
        # Handle File Upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            filepath = os.path.join(UPLOAD_FOLDER, f"upload_{int(time.time())}.jpg")
            file.save(filepath)
            
        # Handle Camera (Base64)
        elif 'image' in request.form:
            image_b64 = request.form['image']
            # Remove header if present (data:image/jpeg;base64,...)
            if ',' in image_b64:
                image_b64 = image_b64.split(',')[1]
            
            image_bytes = base64.b64decode(image_b64)
            filepath = os.path.join(UPLOAD_FOLDER, f"cam_{int(time.time())}.jpg")
            with open(filepath, "wb") as f:
                f.write(image_bytes)
        else:
            return jsonify({'error': 'No image provided'}), 400

        # Process
        result_img, total, fresh, rotten = process_image(filepath)
        
        if result_img is None:
             return jsonify({'error': 'Failed to process image'}), 500

        # Save Result
        filename = f"result_{int(time.time())}.jpg"
        result_path = os.path.join(RESULT_FOLDER, filename)
        cv2.imwrite(result_path, result_img)
        
        return jsonify({
            'total': total,
            'fresh': fresh,
            'rotten': rotten,
            'image_url': f"/static/results/{filename}"
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
