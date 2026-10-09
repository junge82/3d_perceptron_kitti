import os
import numpy as np

def parse_kitti_calib_file(file_path):
    """Parses a KITTI text calibration file into a dictionary of numpy arrays."""
    calib_data = {}
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing calibration file: {file_path}")
    print(f"Parsing calibration file: {file_path}")
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip() or ':' not in line:
                continue
            key, val = line.split(':', 1)
            # Convert text numbers to a flat numpy float array
            numbers_only = []
            for word in val.strip().split():
                try:
                    numbers_only.append(float(word))
                except ValueError:
                    continue # Skip words like "2011_09_26" or "calib_time"
        
            calib_data[key.strip()] = np.array(numbers_only)
    return calib_data

def get_velo_to_image_projection(kitti_dir, cam_id="02"):
    """
    Computes the full 3D Velodyne LiDAR to 2D rectified Image projection matrix.
    cam_id: "02" for left color camera, "03" for right color camera.
    """
    # Define file paths based on standard KITTI raw data tree
    file_velo_to_cam = os.path.join(kitti_dir, "calib_velo_to_cam.txt")    
    file_cam_to_cam = os.path.join(kitti_dir, "calib_cam_to_cam.txt")
    print(f"Using calibration files:\n - {file_velo_to_cam}\n - {file_cam_to_cam}")
    
    # 1. Parse calib_velo_to_cam.txt (Extrinsics)
    ext_data = parse_kitti_calib_file(file_velo_to_cam)
    R_velo_to_cam = ext_data["R"].reshape(3, 3)
    T_velo_to_cam = ext_data["T"].reshape(3, 1)
    
    # Create the 4x4 homogeneous transformation matrix: T_velo_ref_cam
    T_velo_to_ref = np.eye(4)
    T_velo_to_ref[0:3, 0:3] = R_velo_to_cam
    T_velo_to_ref[0:3, 3:4] = T_velo_to_cam

    # 2. Parse calib_cam_to_cam.txt (Intrinsics & Rectification)
    cam_data = parse_kitti_calib_file(file_cam_to_cam)
    
    # R_rect_00 aligns the unrectified reference camera to the rectified plane
    R_rect_00 = cam_data["R_rect_00"].reshape(3, 3)
    
    # Convert R_rect_00 to a 4x4 homogeneous matrix expansion
    R_rect_homo = np.eye(4)
    R_rect_homo[0:3, 0:3] = R_rect_00
    
    # P_rect_xx maps 3D rectified coordinates onto the 2D image plane
    P_rect_cam = cam_data[f"P_rect_{cam_id}"].reshape(3, 4)

    # 3. Chain the transformations together
    # Matrix order (Right to Left): Velo point -> Ref Cam -> Rectified Cam -> Image Pixel
    proj_matrix = np.dot(P_rect_cam, np.dot(R_rect_homo, T_velo_to_ref))
    
    return proj_matrix

def project_velo_to_pixel(point_3d, proj_matrix):
    """
    Projects a 3D Velodyne point [X, Y, Z] to [u, v] image pixel space.
    """
    # Convert point to homogeneous coordinates [X, Y, Z, 1]
    point_homo = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    
    # Matrix multiplication: 3x4 matrix * 4x1 vector = 3x1 vector
    pixel_homo = np.dot(proj_matrix, point_homo)
    
    # Normalize by the depth coordinate (Z channel in camera space)
    z_cam = pixel_homo[2]
    if z_cam <= 0:
        return None  # Point is located behind the camera plane
        
    u = pixel_homo[0] / z_cam
    v = pixel_homo[1] / z_cam
    
    return int(round(u)), int(round(v)), z_cam

# --- Example Driver Execution Verification ---
if __name__ == "__main__":
    # Point this to your actual unzipped day directory containing the calib files
    kitti_dir = "/mnt/fastDisk/kitti3d/raw_data_downloader/2011_09_26/"
    
    try:
        # Generate our final 3x4 transformation matrix
        M_projection = get_velo_to_image_projection(kitti_dir, cam_id="02")
        print("Successfully generated 3x4 Projection Matrix:\n", M_projection)
        
        # Test Point: 5 meters directly in front of the car, slightly left and down
        # KITTI Lidar frame: +X = forward, +Y = left, +Z = up
        sample_point = [5.0, 0.5, -0.2] 
        
        result = project_velo_to_pixel(sample_point, M_projection)
        if result:
            u, v, depth = result
            print(f"\n3D LiDAR Point {sample_point} projects to:")
            print(f" -> Pixel Coordinate (u, v): ({u}, {v})")
            print(f" -> Actual Depth Distance: {depth:.2f} meters")
            
    except FileNotFoundError as e:
        print(f"Setup Error: {e}")
        print("Please verify your file placement paths directory topology layout config.")
