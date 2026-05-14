import torch
from ultralytics import YOLO

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

model = YOLO('yolov8n-obb.pt')

results = model.train(
    data='./yolo_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=4,
    device='cpu',
    amp=False,
    degrees=180,
    mosaic=1.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.5,
    fliplr=0.5,
    project='./runs',
    name='tube_obb',
    exist_ok=True
)

print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"Weights saved to: {results.save_dir}/weights/best.pt")
