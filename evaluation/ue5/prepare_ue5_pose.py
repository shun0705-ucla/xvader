#!/usr/bin/env python3
# relative_pose_exo_to_ego.py
import argparse, os, math
import numpy as np
from pathlib import Path

# ---------- Quaternion helpers (x,y,z,w) ----------
def q_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Zero-norm quaternion")
    return q / n

def q_to_R(q):
    # q = [x, y, z, w]
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy+zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),       1 - 2*(xx+zz),     2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),       1 - 2*(xx+yy)]
    ], dtype=float)
    return R

def R_to_q(R):
    # returns [x,y,z,w], robust branch
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
            w = (R[2,1] - R[1,2]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
            w = (R[0,2] - R[2,0]) / s
        else:
            s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
            w = (R[1,0] - R[0,1]) / s
    q = np.array([x, y, z, w], dtype=float)
    return q_normalize(q)

def SE3_from_tq(t, q):
    R = q_to_R(q_normalize(q))
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T

def SE3_inv(T):
    R = T[:3,:3]
    t = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3,3]  = -R.T @ t
    return Ti

def SE3_to_tq(T):
    R = T[:3,:3]
    t = T[:3,3]
    q = R_to_q(R)
    return t, q

# ---------- I/O ----------
def load_pose_rows(path):
    arr = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 7:
                raise ValueError(f"Each row must have 7 values (tx ty tz qx qy qz qw). Found {len(vals)} in: {path}")
            t = vals[0:3]
            q = vals[3:7]  # [x,y,z,w]
            arr.append((t, q))
    return arr

def save_row(out_path, data, fmt):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if fmt == "quat":
        # tx ty tz qx qy qz qw
        t, q = data
        line = "{:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e}\n".format(
            t[0], t[1], t[2], q[0], q[1], q[2], q[3]
        )
        with open(out_path, "w") as f:
            f.write(line)
    elif fmt == "matrix":
        T = data  # 4x4
        np.savetxt(out_path, T, fmt="%.9e")
    else:
        raise ValueError("Unknown output_format")
    
def convert_ue_to_opencv(t_ue_cm, q_ue):
    # Fixed UE -> OpenCV change-of-basis matrix S (see mapping above)
    S   = [[0, 1, 0],
           [0, 0,-1],
           [1, 0, 0]]
    ST  = [[S[0][0], S[1][0], S[2][0]],
           [S[0][1], S[1][1], S[2][1]],
           [S[0][2], S[1][2], S[2][2]]]
    # rotations: R_cv = S * R_ue * S^T
    def mat3_mul(A,B):
        return [[sum(A[r][k]*B[k][c] for k in range(3)) for c in range(3)] for r in range(3)]
    def mat3_vec(A,v):
        return (
            A[0][0]*v[0] + A[0][1]*v[1] + A[0][2]*v[2],
            A[1][0]*v[0] + A[1][1]*v[1] + A[1][2]*v[2],
            A[2][0]*v[0] + A[2][1]*v[1] + A[2][2]*v[2],
        )
    
    R_ue = q_to_R(q_ue)
    R_tmp = mat3_mul(S, R_ue)
    R_cv  = mat3_mul(R_tmp, ST)
    q_cv  = R_to_q(R_cv)

    # translations: t_cv (meters) = (S * t_ue_cm) / 100
    t_cv_m = tuple(x/100.0 for x in mat3_vec(S, t_ue_cm))
    return t_cv_m, q_cv

# ---------- Main ----------
def main():
    exo_rows = load_pose_rows(Path("./ue5/WildWest_test_exo/pose_exo.txt"))
    ego_rows = load_pose_rows(Path("./ue5/WildWest_test_ego/pose_ego.txt"))
    out_dir = Path("./ue5/WildWest_test_ego/gt_camera")
    out_dir.mkdir(parents=True, exist_ok=True)
    poses_are = "cam_to_world"
    output_format = "quat"

    if len(exo_rows) != len(ego_rows):
        raise ValueError(f"Row count mismatch: exo={len(exo_rows)} vs ego={len(ego_rows)}")

    for i, ((t_exo, q_exo), (t_ego, q_ego)) in enumerate(zip(exo_rows, ego_rows)):
        # Build SE(3) for each
        T_exo = SE3_from_tq(t_exo, q_exo)
        T_ego = SE3_from_tq(t_ego, q_ego)

        # We want T_rel that maps points from EXO frame to EGO frame.
        # If inputs are Twc (cam->world):   T_rel = (T_ego)^-1 @ T_exo
        # If inputs are Tcw (world->cam):   T_rel = T_ego @ (T_exo)^-1
        if poses_are == "cam_to_world":
            T_rel = SE3_inv(T_ego) @ T_exo
        else:  # world_to_cam
            T_rel = T_ego @ SE3_inv(T_exo)

        out_path = os.path.join(out_dir, f"{i:06d}.txt")
        if output_format == "quat":
            t_rel, q_rel = SE3_to_tq(T_rel)
            t_cv_m, q_cv = convert_ue_to_opencv(t_rel, q_rel)
            save_row(out_path, (t_cv_m, q_cv), "quat")
        else:
            save_row(out_path, T_rel, "matrix")

    print(f"Done. Wrote {len(exo_rows)} files to {out_dir}")

if __name__ == "__main__":
    main()