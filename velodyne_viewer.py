import numpy as np
import open3d as o3d
import sys

def view_velodyne_bin(bin_path):
    # 1. Read the binary file. Velodyne files use float32 numbers.
    # Each point contains 4 attributes: X, Y, Z, and Intensity/Reflectance.
    try:
        point_cloud_raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    except FileNotFoundError:
        print(f"Error: The file '{bin_path}' was not found.")
        return
    except ValueError:
        print("Error: File shape doesn't match standard (N, 4) Velodyne layout.")
        return

    # 2. Extract spatial coordinates (X, Y, Z)
    xyz = point_cloud_raw[:, :3]
    
    # 3. Extract reflectance/intensity (optional, can be used for coloring)
    intensity = point_cloud_raw[:, 3]

    # 4. Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # 5. Optional: Normalize intensity to map colors (grayscale effect)
    if len(intensity) > 0:
        # Normalize between 0 and 1
        normalized_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity) + 1e-8)
        # Tile into 3 channels (RGB) for grayscale coloring
        colors = np.vstack((normalized_intensity, normalized_intensity, normalized_intensity)).T
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # 6. Render the Interactive Window
    print(f"Displaying {len(xyz)} points. Close window to exit.")
    o3d.visualization.draw_geometries([pcd], 
                                      window_name="Velodyne .bin LiDAR Viewer",
                                      width=1024, 
                                      height=768)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bin_viewer.py <path_to_file.bin>")
    else:
        view_velodyne_bin(sys.argv[1])
