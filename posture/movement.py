"""Running vs non-running from joint angles in a single frame.

A person is called "running" when at least one elbow is bent below
`ARM_THRESHOLD` degrees AND at least one knee is bent below `LEG_THRESHOLD`.
Both legs visible and no running pattern -> "non-running". Otherwise unknown.
"""
import numpy as np

from .keypoints import (L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
                        R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
                        visible)

CONF = 0.3
ARM_THRESHOLD = 130
LEG_THRESHOLD = 160

RUNNING, NON_RUNNING, UNKNOWN = "running", "non-running", "unknown"

ARMS = {"left": (L_SHOULDER, L_ELBOW, L_WRIST), "right": (R_SHOULDER, R_ELBOW, R_WRIST)}
LEGS = {"left": (L_HIP, L_KNEE, L_ANKLE), "right": (R_HIP, R_KNEE, R_ANKLE)}


def joint_angle(a, b, c):
    """Angle at b (degrees) between segments b->a and b->c. 180 = straight limb."""
    ba, bc = np.asarray(a, float) - b, np.asarray(c, float) - b
    cos = ba @ bc / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _angles(kpts, conf, limbs, threshold):
    return {side: joint_angle(*(kpts[j] for j in joints))
            for side, joints in limbs.items() if visible(conf, joints, threshold)}


def classify_movement(kpts, conf, threshold=CONF,
                      arm_threshold=ARM_THRESHOLD, leg_threshold=LEG_THRESHOLD):
    """Return (label, details dict of the measured angles)."""
    kpts = np.asarray(kpts, dtype=float)
    conf = np.asarray(conf, dtype=float)
    arms = _angles(kpts, conf, ARMS, threshold)
    legs = _angles(kpts, conf, LEGS, threshold)
    details = {"arms": arms, "legs": legs}

    arm_bent = any(a < arm_threshold for a in arms.values())
    leg_bent = any(a < leg_threshold for a in legs.values())
    if arm_bent and leg_bent:
        return RUNNING, details
    if len(legs) == 2:
        return NON_RUNNING, details
    return UNKNOWN, details
