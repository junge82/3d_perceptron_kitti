#!/usr/bin/env python3
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO to Optimized ONNX Exporter")
    parser.add_argument("--model", type=str, default="yolov8n", help="Model name (yolov8n) or path to .pt file")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the exported ONNX model")
    parser.add_argument("--img-size", type=int, default=640, help="Spatial resolution image size input")
    args = parser.parse_args()

    # 1. Delay heavy library initialization until arguments are validated
    print("--> Initializing core ML frameworks...")
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[CRITICAL] Missing dependencies. Run: pip install ultralytics torch")
        sys.exit(1)

    # 2. Determine paths
    os.makedirs(args.output_dir, exist_ok=True)
    if args.model.endswith(".pt"):
        pt_path = args.model
        base_name = os.path.splitext(os.path.basename(pt_path))[0]
    else:
        pt_path = f"{args.model}.pt"
        base_name = args.model

    final_onnx_path = os.path.join(args.output_dir, f"{base_name}.onnx")
    print(f"[INFO] Source Weights : {pt_path}")
    print(f"[INFO] Target Export : {final_onnx_path}")

    # 3. Load model structure into memory
    print("\n--> Loading weights matrix...")
    try:
        model = YOLO(pt_path)
    except Exception as e:
        print(f"[CRITICAL] Failed to load YOLO weights: {e}")
        sys.exit(1)

    # 4. Execute the hardware-optimized ONNX serialization pipeline
    print("\n--> Compiling model into static tensor graph...")
    try:
        # We explicitly enforce parameters required for high-speed ONNX Runtime execution
        exported_output = model.export(
            format="onnx",
            imgsz=[args.img_size, args.img_size],
            dynamic=False,        # Fix shapes to maximize memory allocation speed on the GPU
            simplify=True,        # Fuse redundant layers to minimize graph complexity
            opset=14,             # Best compatibility and performance match for ONNX Runtime 1.18+
            half=False            # Export in FP32 (recommended for standard CUDAExecutionProvider)
        )

        # Ultralytics exports to the same directory as the source .pt file by default
        default_export_path = pt_path.replace(".pt", ".onnx")

        if os.path.exists(default_export_path) and default_export_path != final_onnx_path:
            import shutil
            shutil.move(default_export_path, final_onnx_path)

        print("\n" + "=" * 60)
        print(" SUCCESS: ONNX Graph successfully compiled and optimized! ")
        print(f" Saved artifact path: {final_onnx_path} ")
        print("=" * 60)

    except Exception as e:
        print(f"\n[CRITICAL] Export workflow failure: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
