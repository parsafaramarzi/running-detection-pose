"""COCO-17 keypoint layout used by YOLO pose models, plus a visibility helper."""
import numpy as np

NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
IDX = {name: i for i, name in enumerate(NAMES)}

L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

BODY_JOINTS = list(range(5, 17))  # the 12 joints below the face

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


def visible(conf, joints, threshold):
    """True when every joint in `joints` has confidence above `threshold`."""
    conf = np.asarray(conf)
    return all(conf[j] > threshold for j in joints)
