from ultralytics import YOLO

# Load your baseline PyTorch weights
model = YOLO('yolov8n.pt')

# ⚡ CRITICAL FIX: Set baseline size to KITTI dimensions AND enable dynamic axes
model.export(
    format='onnx',
    imgsz=[384, 1248], # Baseline height & width (multiples of 32 near KITTI)
    dynamic=True       # Allows small pixel shifts around this size
)