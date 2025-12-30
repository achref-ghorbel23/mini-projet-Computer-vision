# =====================================================
# MULTI TOMATO DETECTION + CLASSIFICATION
# LOGIQUE :
# YOLO PERSONNALISÉ S'IL EXISTE
# SINON YOLOv8 STANDARD
# =====================================================

import cv2
import numpy as np
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# =========================
# 1️⃣ Paths
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, "tomato_fresh_rotten_model.h5")
CUSTOM_TOMATO_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
YOLO_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

# =========================
# 2️⃣ Chargement modèles
# =========================

print("Chargement des modèles...")

try:
    classifier = load_model(CLASSIFIER_MODEL_PATH)

    # 🔥 LOGIQUE DEMANDÉE
    if os.path.exists(CUSTOM_TOMATO_MODEL_PATH):
        print(f"✔ Modèle YOLO personnalisé trouvé : {CUSTOM_TOMATO_MODEL_PATH}")
        detector = YOLO(CUSTOM_TOMATO_MODEL_PATH)
        USING_CUSTOM_MODEL = True
    else:
        print(f"⚠ Modèle personnalisé introuvable → YOLOv8 standard")
        detector = YOLO(YOLO_PATH)
        USING_CUSTOM_MODEL = False

    print("Modèles chargés avec succès")

except Exception as e:
    print(f"Erreur chargement modèles : {e}")
    raise e

# =========================
# 3️⃣ Classification Fresh / Rotten
# =========================

def classify_tomato(image_crop):
    img = cv2.resize(image_crop, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = classifier.predict(img, verbose=0)[0][0]

    if pred > 0.5:
        return "Rotten", float(pred)
    else:
        return "Fresh", float(1 - pred)

# =========================
# 4️⃣ Resize affichage
# =========================

def resize_for_display(img, max_width=1000):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# =========================
# 5️⃣ Charger image
# =========================

image_path = "test2.png"
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Image introuvable")

# =========================
# 6️⃣ Détection (UN SEUL YOLO)
# =========================

if USING_CUSTOM_MODEL:
    results = detector(
        image,
        conf=0.40,
        iou=0.20,
        agnostic_nms=True
    )[0]
else:
    results = detector(
        image,
        conf=0.15,
        iou=0.7,
        agnostic_nms=True
    )[0]

# =========================
# 7️⃣ Traitement des détections
# =========================

tomato_count = 0
fresh_count = 0
rotten_count = 0

for box in results.boxes:

    cls_id = int(box.cls[0])
    label = detector.names[cls_id]

    # 🔵 Filtrage UNIQUEMENT pour YOLOv8 standard
    if not USING_CUSTOM_MODEL:
        if label not in ["apple", "orange", "sports ball", "ball"]:
            continue

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    w, h = x2 - x1, y2 - y1

    # Filtre bruit / distance
    if w < 30 or h < 30:
        continue

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
        f"{state} {confidence:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

# =========================
# 8️⃣ Résumé
# =========================

model_used = "YOLO CUSTOM" if USING_CUSTOM_MODEL else "YOLOv8 STANDARD"

cv2.putText(
    image,
    f"{model_used} | Total: {tomato_count} | Fresh: {fresh_count} | Rotten: {rotten_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

# =========================
# 9️⃣ Affichage
# =========================

display_img = resize_for_display(image)
cv2.imshow("Tomato Detection", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


