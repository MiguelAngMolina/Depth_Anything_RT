import os
import cv2
import time
import numpy as np
import torch
import matplotlib
from depth_anything_v2.dpt import DepthAnythingV2

# Configuración inicial
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
ENCODER = 'vits'  # Puedes cambiar a vits, vitb o vitg
INPUT_SIZE = 518

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Cargar el modelo
print("🔄 Cargando modelo...")
depth_model = DepthAnythingV2(**model_configs[ENCODER])
depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{ENCODER}.pth', map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()
print("✅ Modelo cargado.")

# URL de la cámara
url = "http://10.14.21.99:8080/video"
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

print("📷 Cámara activa. Presiona ESC para salir.")
capture_count = 0
last_capture_time = time.time()
width, height = 600, 320
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error leyendo la cámara.")
        break

    frame = cv2.flip(frame, 1)
    resized_frame = cv2.resize(frame, (width, height))
    cv2.imshow("Cámara en vivo", resized_frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

    if time.time() - last_capture_time >= 5:
        print(f"📸 Captura #{capture_count}")
        capture = frame.copy()

        # Inferencia de profundidad
        depth = depth_model.infer_image(capture[:, :, ::-1], INPUT_SIZE)
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_colored = (cmap(depth_norm.astype(np.uint8))[:, :, :3] * 255).astype(np.uint8)
        depth_resized = cv2.resize(depth_colored, (width, height))

        cv2.imshow("Profundidad", depth_resized)
        capture_count += 1
        last_capture_time = time.time()

cap.release()
cv2.destroyAllWindows()
