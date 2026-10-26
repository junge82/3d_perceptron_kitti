import numpy as np
import scipy.spatial.transform as st

# 1. Másoljuk be a te calib_velo_to_cam.txt fájlod pontos adatait
R_raw = np.array([
    7.533745e-03, -9.999714e-01, -6.166020e-04,
    1.480249e-02,  7.280733e-04, -9.998902e-01,
    9.998621e-01,  7.523790e-03,  1.480755e-02
]).reshape(3, 3)

T_raw = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])

# 2. Számoljuk ki a ROS kompatibilis invers transzformációt (Cam -> Velo)
R_cam_to_velo = R_raw.T
t_cam_to_velo = np.dot(-R_raw.T, T_raw)

# 3. Alakítsuk át a rotációs mátrixot ROS [x, y, z, w] kvaternióvá
r = st.Rotation.from_matrix(R_cam_to_velo)
q = r.as_quat()  # SciPy alapértelmezetten [x, y, z, w] sorrendet ad vissza

# 4. Írassuk ki a launch fájlba illeszthető pontos argumentumokat
print("\n=== MÁSOLD BE EZEKET AZ UTOLSÓ LAUNCH FÁJLODBA ===\n")
print(f"'--x', '{t_cam_to_velo[0]:.4f}', '--y', '{t_cam_to_velo[1]:.4f}', '--z', '{t_cam_to_velo[2]:.4f}',")
print(f"'--qx', '{q[0]:.4f}', '--qy', '{q[1]:.4f}', '--qz', '{q[2]:.4f}', '--qw', '{q[3]:.4f}',")
print("\n==================================================\n")
