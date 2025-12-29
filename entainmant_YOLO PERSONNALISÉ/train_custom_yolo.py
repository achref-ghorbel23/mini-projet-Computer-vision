from ultralytics import YOLO

# 1. Charger le modèle de base (YOLOv8 Nano)
# On part de "yolov8n.pt" pour faire du Transfer Learning
model = YOLO('yolov8n.pt')

# 2. Lancer l'entraînement
# data='archive/data.yaml' : le fichier qui contient les chemins vers vos images et les noms des classes
# epochs=50 : nombre de fois où le modèle va voir toutes les images (ajustez selon besoin)
# imgsz=640 : taille des images pour l'entraînement (standard YOLO)
print("🚀 Démarrage de l'entraînement...")

results = model.train(
    data='archive/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='custom_tomato_model' # Nom du dossier de sortie dans "runs/detect/"
)

print("✅ Entraînement terminé !")
print(f"Le meilleur modèle est sauvegardé ici : {results.save_dir}/weights/best.pt")
