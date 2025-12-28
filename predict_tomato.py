# =====================================================
# MULTI TOMATO DETECTION + CLASSIFICATION (FRESH / ROTTEN)
# =====================================================

import cv2
import numpy as np
import json
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# =========================
# 1️⃣ Charger les modèles
# =========================

print("Chargement des modèles...")

# Modèle de détection (YOLOv8)
detector = YOLO("yolov8n.pt")

# Modèle de classification (TON modèle)
classifier = load_model("tomato_fresh_rotten_model.h5")

print("Modèles chargés avec succès")

# =========================
# 2️⃣ Charger le mapping
# =========================

with open("class_mapping.json", "r") as f:
    class_mapping = json.load(f)

# Inversion : {0: 'Fresh', 1: 'Rotten'}
class_names = {v: k for k, v in class_mapping.items()}

print("Mapping des classes :", class_names)

# =========================
# 3️⃣ Fonction classification
# =========================

def classify_tomato(image_crop):
    """
    image_crop : image BGR (OpenCV)
    return : label, confidence
    """
    img = cv2.resize(image_crop, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = classifier.predict(img, verbose=0)[0][0]

    if prediction > 0.5:
        return "Rotten", float(prediction)
    else:
        return "Fresh", float(1 - prediction)

# =========================
# 4️⃣ Charger l'image
# =========================

image_path = "ABV.jpg"
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Image introuvable : vérifie le chemin")

print("Image chargée :", image_path)

# =========================
# 5️⃣ Détection des tomates
# =========================

results = detector(image)[0]

tomato_count = 0
fresh_count = 0
rotten_count = 0

for box in results.boxes:

    cls_id = int(box.cls[0])
    label = detector.names[cls_id]

    # YOLO n'a pas la classe "tomato"
    # On accepte fruits proches visuellement
    if label not in ["apple", "orange", "sports ball", "ball"]:
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        continue

    tomato_count += 1

    state, confidence = classify_tomato(crop)

    if state == "Fresh":
        color = (0, 255, 0)
        fresh_count += 1
    else:
        color = (0, 0, 255)
        rotten_count += 1

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        image,
        f"{state} ({confidence:.2f})",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

# =========================
# 6️⃣ Résumé à l'écran
# =========================

cv2.putText(
    image,
    f"Total: {tomato_count} | Fresh: {fresh_count} | Rotten: {rotten_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

# =========================
# 7️⃣ Affichage final
# =========================

cv2.imshow("Multi Tomato Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
