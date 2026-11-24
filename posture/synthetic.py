"""A synthetic standing skeleton, used by tests and the sensitivity analysis."""
import numpy as np

# COCO-17 order, image coordinates (y down), pixels. Height about 190 px.
_STANDING = np.array([
    (0, -25), (-5, -30), (5, -30), (-12, -25), (12, -25),   # nose, eyes, ears
    (-20, 0), (20, 0),                                      # shoulders
    (-25, 35), (25, 35), (-25, 70), (25, 70),               # elbows, wrists
    (-15, 80), (15, 80),                                    # hips
    (-15, 125), (15, 125), (-15, 170), (15, 170),           # knees, ankles
], dtype=float)


def standing_pose():
    """(keypoints[17, 2], confidences[17]) of an upright person, all visible."""
    return _STANDING.copy(), np.ones(17)


def rotate(kpts, degrees, about=(0.0, 85.0)):
    """Rotate keypoints about a point. 0 = upright, 90 = lying, 180 = upside down."""
    t = np.radians(degrees)
    r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    c = np.asarray(about)
    return (np.asarray(kpts) - c) @ r.T + c


def hide(conf, joints):
    """Copy of `conf` with the given joint indices set to zero confidence."""
    out = np.asarray(conf, dtype=float).copy()
    out[list(joints)] = 0.0
    return out
