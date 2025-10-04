# src/gait_viz.py
import os, cv2, numpy as np, pandas as pd
from scipy.spatial import distance
from ultralytics import YOLO
from ultralytics.utils import LOGGER

LOGGER.setLevel("WARNING")
LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX = 15, 16

def safe_kpt(person_17x2, idx):
    if idx >= person_17x2.shape[0]: return None
    x, y = person_17x2[idx]
    if (x == 0 and y == 0) or np.isnan(x) or np.isnan(y): return None
    return np.array([float(x), float(y)], dtype=float)

def smooth(prev, curr, alpha=0.25):
    if prev is None or curr is None: return curr
    return (1 - alpha) * prev + alpha * curr

def choose_person_by_proximity(K_all, prev_ankle, fallback_conf=None):
    if K_all.shape[0] == 0: return None
    if prev_ankle is None:
        return int(np.argmax(fallback_conf)) if fallback_conf is not None else 0
    d = []
    for i in range(K_all.shape[0]):
        pt = K_all[i, LEFT_ANKLE_IDX]
        d.append(1e9 if (pt[0] == 0 and pt[1] == 0) else np.linalg.norm(prev_ankle - pt))
    return int(np.argmin(d))

def run_pose_on_video(
    in_path, out_video="outputs/pose_estimation_output.mp4",
    out_csv="outputs/gait_metrics.csv",
    model_name="yolov8s-pose.pt",
    imgsz=640, conf=0.25, iou=0.7, smooth_alpha=0.25
):
    os.makedirs("outputs", exist_ok=True)
    model = YOLO(model_name)

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

    gait_rows = []
    frame_count = 0
    ema_left = ema_right = prev_left = prev_right = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        annotated = results[0].plot()

        kpts = results[0].keypoints
        if (kpts is None) or (kpts.xy is None):
            out.write(annotated); frame_count += 1; continue

        K = kpts.xy.cpu().numpy()  # (N,17,2)
        if K.shape[0] == 0:
            out.write(annotated); frame_count += 1; continue

        confs = (kpts.conf.cpu().numpy().mean(axis=1)
                    if kpts.conf is not None else np.ones(K.shape[0]))
        idx = choose_person_by_proximity(K, prev_left, confs)
        person = K[idx]

        left, right = safe_kpt(person, LEFT_ANKLE_IDX), safe_kpt(person, RIGHT_ANKLE_IDX)
        ema_left, ema_right = smooth(ema_left, left, smooth_alpha), smooth(ema_right, right, smooth_alpha)

        stride_len = None
        if (prev_left is not None) and (ema_left is not None):
            stride_len = float(distance.euclidean(prev_left, ema_left))

        step_duration = None if stride_len is None else (frame_count / fps)

        gait_rows.append({
            "frame": frame_count,
            "left_x": None if ema_left is None else float(ema_left[0]),
            "left_y": None if ema_left is None else float(ema_left[1]),
            "right_x": None if ema_right is None else float(ema_right[0]),
            "right_y": None if ema_right is None else float(ema_right[1]),
            "stride_px": stride_len,
            "step_duration_s": step_duration
        })

        prev_left = ema_left if ema_left is not None else prev_left
        prev_right = ema_right if ema_right is not None else prev_right

        if ema_left is not None:
            cv2.circle(annotated, (int(ema_left[0]), int(ema_left[1])), 4, (0,255,0), -1, lineType=cv2.LINE_AA)
        if ema_right is not None:
            cv2.circle(annotated, (int(ema_right[0]), int(ema_right[1])), 4, (255,0,0), -1, lineType=cv2.LINE_AA)

        out.write(annotated)
        frame_count += 1

    cap.release(); out.release()

    df = pd.DataFrame(gait_rows)
    df.to_csv(out_csv, index=False)

    # naive cadence estimate (steps/min) from nonzero stride frames
    valid = df["stride_px"].dropna()
    cadence_est = None
    if len(valid) > 3:
        duration_s = df["frame"].max() / (fps or 25.0)
        steps = (valid > np.percentile(valid, 60)).sum()  # crude threshold
        cadence_est = 60.0 * steps / max(duration_s, 1e-6)

    return out_video, out_csv, cadence_est