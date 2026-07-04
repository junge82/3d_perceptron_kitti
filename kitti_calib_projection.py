import os
import sys
import numpy as np
import cv2

def load_kitti_calib(calib_path):
    """Parses KITTI calibration file to extract P2 and Tr_velo_to_cam matrices."""
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            key, value = line.split(':', 1)
            calib[key.strip()] = np.fromstring(value.strip(), sep=' ')
            
    # P2 matrix: Shape (3, 4) -> Camera 2 intrinsic/projection matrix
    P2 = calib['P2'].reshape(3, 4)
    
    # Tr_velo_to_cam: Shape (3, 4) -> Rigid transformation from Velodyne to Camera reference
    Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
    # Convert to a (4, 4) homogeneous transformation matrix
    Tr_velo_to_cam = np.vstack((Tr_velo_to_cam, [0, 0, 0, 1]))
    
    # R0_rect: Shape (3, 3) -> Rectification matrix
    R0_rect = calib['R0_rect'].reshape(3, 3)
    # Convert to a (4, 4) homogeneous matrix
    R0_rect_homo = np.eye(4)
    R0_rect_homo[:3, :3] = R0_rect
    
    return P2, Tr_velo_to_cam, R0_rect_homo

def project_velo_to_image2(bin_path, image_path, calib_path, label_path=None):
    # 1. Load data files
    if not (os.path.exists(bin_path) and os.path.exists(image_path) and os.path.exists(calib_path)):
        print("Error: One or more data file paths do not exist.")
        return

    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    # Load LiDAR data (X, Y, Z, Reflectance)
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pts_3d = scan[:, :3] # Keep only X, Y, Z

    # Load calibration matrices
    P2, Tr_velo_to_cam, R0_rect = load_kitti_calib(calib_path)

    # 2. Filter out points behind the camera (X is forward in Velodyne coordinates)
    front_indices = pts_3d[:, 0] > 0
    pts_3d = pts_3d[front_indices]

    # 3. Convert LiDAR points to Homogeneous coordinates (N, 4)
    pts_3d_homo = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))

    # 4. Transform from Velodyne space to Camera Ref space, then apply Rectification
    # Context formula: P_cam = R0_rect * Tr_velo_to_cam * P_velo
    pts_cam = np.dot(R0_rect, np.dot(Tr_velo_to_cam, pts_3d_homo.T)).T

    # 5. Project 3D Camera space points onto 2D image plane using P2
    pts_2d_homo = np.dot(P2, pts_cam.T).T
    
    # Normalize by the depth coordinate (Z channel) to get pixel spaces (u, v)
    pts_2d = pts_2d_homo[:, :2] / pts_2d_homo[:, 2:3]

    # 6. Filter out points that fall outside the image field of view boundaries
    valid_indices = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
                    (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    
    pts_2d = pts_2d[valid_indices]
    depths = pts_cam[valid_indices, 2] # Depth values relative to camera frame

    # 7. Render depth points onto image array using a color map
    max_depth = 40.0 # Clip maximum visual distance to 40 meters
    for i in range(len(pts_2d)):
        x, y = int(pts_2d[i, 0]), int(pts_2d[i, 1])
        depth = depths[i]
        
        # Color spectrum scaling based on distance (Closer = Red, Further = Blue)
        normalized_depth = min(depth / max_depth, 1.0)
        color_value = int(normalized_depth * 255)
        color = cv2.applyColorMap(np.array([[color_value]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
        color = (int(color[0]), int(color[1]), int(color[2]))

        cv2.circle(img, (x, y), 2, color, -1)

    # Display final projection image window
    cv2.imshow("LiDAR Points Projected to image_2 Plane", img)
    # If label file provided, draw 2D bounding boxes and class names
    if label_path is not None and os.path.exists(label_path):
        with open(label_path, 'r') as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                cls = parts[0]
                # KITTI label format: class, truncation, occlusion, alpha, bbox (4 values), ...
                try:
                    bbox = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                except ValueError:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                print(f"Drawing {cls} bounding box: ({x1}, {y1}), ({x2}, {y2})")
                # draw rectangle and put class text
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, cls, (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        # redisplay updated image with labels
        cv2.imshow("LiDAR Points Projected to image_2 Plane", img)
    print("Window active. Press 'q' or 'ESC' to exit application.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example paths relative to KITTI tracking or object dataset structure
    # Adjust files targeting your specific dataset sequence tracking folder
    BIN_FILE = "/mnt/fastDisk/kitti3d/kitti_object/training/velodyne/000000.bin"
    IMG_FILE = "/mnt/fastDisk/kitti3d/kitti_object/training/image_2/000000.png"
    CALIB_FILE = "/mnt/fastDisk/kitti3d/kitti_object/training/calib/000000.txt"
    LABEL_FILE = "/mnt/fastDisk/kitti3d/kitti_object/training/label_2/000000.txt"  # Optional
    
    project_velo_to_image2(BIN_FILE, IMG_FILE, CALIB_FILE, label_path=LABEL_FILE)
