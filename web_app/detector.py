import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
CLASSIFIER_MODEL_PATH = os.path.join(PARENT_DIR, "tomato_fresh_rotten_model.h5")
YOLO_PATH = os.path.join(PARENT_DIR, "yolov8n.pt")
CUSTOM_TOMATO_MODEL_PATH = os.path.join(PARENT_DIR, "best.pt") # Path for the custom model

print(f"Loading models from: {PARENT_DIR}")

# Load Models
try:
    classifier = load_model(CLASSIFIER_MODEL_PATH)
    
    # Check if custom tomato model exists
    if os.path.exists(CUSTOM_TOMATO_MODEL_PATH):
        print(f"Found custom tomato model: {CUSTOM_TOMATO_MODEL_PATH}")
        detector = YOLO(CUSTOM_TOMATO_MODEL_PATH)
        USING_CUSTOM_MODEL = True
    else:
        print(f"Custom model not found. Using standard YOLOv8: {YOLO_PATH}")
        detector = YOLO(YOLO_PATH)
        USING_CUSTOM_MODEL = False
        
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    raise e

def is_orange_color(image_crop):
    """
    Check if the dominant color of the crop is orange.
    Returns True if orange, False otherwise.
    """
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
    
    # Orange range in HSV (approximate)
    # Hue: 10-25 (OpenCV uses 0-179 for Hue)
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    ratio = cv2.countNonZero(mask) / (image_crop.shape[0] * image_crop.shape[1])
    
    # If more than 30% of the object is orange, it's likely an orange
    return ratio > 0.3

def classify_tomato(image_crop):
    """
    Classify crop as Fresh or Rotten.
    """
    img = cv2.resize(image_crop, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = classifier.predict(img, verbose=0)[0][0]

    if prediction > 0.5:
        return "Rotten", float(prediction)
    else:
        return "Fresh", float(1 - prediction)

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, 0, 0, 0

    # Run YOLOv8
    # If using custom model, we rely on its specific training.
    # We relax iou to 0.30 (from 0.20) to be less aggressive on small distant groups.
    # We keep conf=0.20 to catch the "missing" tomato.
    if USING_CUSTOM_MODEL:
        results = detector(image, conf=0.40, iou=0.20, agnostic_nms=True)[0] 
    else:
        results = detector(image, conf=0.15, iou=0.7, agnostic_nms=True)[0]
    
    tomato_count = 0
    fresh_count = 0
    rotten_count = 0
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = detector.names[cls_id]
        conf = float(box.conf[0])
        print(f"DEBUG: Detected {label} with confidence {conf}")

        # Filter candidates based on model type
        if USING_CUSTOM_MODEL:
            # Assume custom model has classes like 'tomato', 'fresh', 'rotten' or just class 0
            # We accept everything the custom model predicts as a "potential tomato"
            pass 
        else:
            # Standard Model Logic: Look for apples, oranges, balls
            if label not in ["apple", "orange", "sports ball", "ball"]:
                continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # New Filter: Minimum Size for "Far" objects logic
        # If the object is too small (e.g. < 30x30 pixels), it's likely noise or too far to classify correctly.
        # This fixes "detecte faux" for far objects.
        w = x2 - x1
        h = y2 - y1
        if w < 30 or h < 30:
            print(f"DEBUG: Skipping small object {w}x{h}")
            continue

        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue

        # Classify (Fresh/Rotten)
        # We use our separate classifier for consistency, even if the YOLO model has classes.
        # This ensures our Fresh/Rotten logic remains robust.
        state, confidence = classify_tomato(crop)
        
        tomato_count += 1
        if state == "Fresh":
            fresh_count += 1
            color = (0, 255, 0)
        else:
            rotten_count += 1
            color = (0, 0, 255)

        detections.append({
            "box": [x1, y1, x2, y2],
            "label": state,
            "confidence": confidence,
            "color": color
        })

        # Draw on image
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{state} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add Summary Text
    cv2.putText(
        image,
        f"Total: {tomato_count} | Fresh: {fresh_count} | Rotten: {rotten_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    return image, tomato_count, fresh_count, rotten_count
