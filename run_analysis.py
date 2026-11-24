"""Where does the running rule switch? Sweep elbow and knee angles on a synthetic skeleton.

There is no labelled video in this repo, so no accuracy is claimed. This script
shows exactly which joint-angle combinations the rule calls running, and how it
reacts to hidden joints.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posture import movement
from posture.keypoints import L_ANKLE, L_ELBOW, L_KNEE, L_SHOULDER, L_WRIST, R_ANKLE, R_KNEE
from posture.synthetic import hide, standing_pose

OUT = Path("results")
OUT.mkdir(exist_ok=True)


def bent_pose(elbow_deg, knee_deg):
    """Standing skeleton whose LEFT elbow and LEFT knee are bent to the given angles."""
    k, c = standing_pose()
    k = k.copy()
    for joint, end, first in ((L_ELBOW, L_WRIST, L_SHOULDER), (L_KNEE, L_ANKLE, 11)):
        a = np.radians(angle := (elbow_deg if joint == L_ELBOW else knee_deg))
        length = np.linalg.norm(k[end] - k[joint])
        up = (k[first] - k[joint]) / np.linalg.norm(k[first] - k[joint])
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        k[end] = k[joint] + length * (rot @ up)
    return k, c


def grid():
    rows = []
    for elbow in range(45, 176, 10):
        for knee in range(85, 176, 10):
            k, c = bent_pose(elbow, knee)
            label, d = movement.classify_movement(k, c)
            rows.append({"elbow_deg": elbow, "knee_deg": knee, "label": label,
                         "measured_elbow": round(d["arms"]["left"], 1),
                         "measured_knee": round(d["legs"]["left"], 1)})
    return pd.DataFrame(rows)


def occlusion_cases():
    k, c = bent_pose(90, 120)
    cases = {
        "bent arm and leg, all visible": c,
        "same, left wrist hidden": hide(c, [L_WRIST]),
        "same, left ankle hidden": hide(c, [L_ANKLE]),
        "same, both ankles hidden": hide(c, [L_ANKLE, R_ANKLE]),
        "same, both knees hidden": hide(c, [L_KNEE, R_KNEE]),
    }
    return pd.DataFrame([{"case": n, "label": movement.classify_movement(k, cc)[0]}
                         for n, cc in cases.items()])


if __name__ == "__main__":
    g = grid()
    g.to_csv(OUT / "angle_grid.csv", index=False)
    occ = occlusion_cases()
    occ.to_csv(OUT / "occlusion_cases.csv", index=False)

    piv = g.assign(v=g.label.map({"running": 1, "non-running": 0, "unknown": 0.5}))            .pivot(index="knee_deg", columns="elbow_deg", values="v")
    fig, a = plt.subplots(figsize=(8, 4.5))
    a.imshow(piv.values, origin="lower", aspect="auto", cmap="coolwarm", vmin=0, vmax=1,
             extent=[piv.columns.min() - 5, piv.columns.max() + 5, piv.index.min() - 5, piv.index.max() + 5])
    a.axvline(movement.ARM_THRESHOLD, color="k", ls="--")
    a.axhline(movement.LEG_THRESHOLD, color="k", ls="--")
    a.set_xlabel("left elbow angle (degrees, 180 = straight)")
    a.set_ylabel("left knee angle (degrees)")
    a.set_title("Red = running, blue = non-running")
    fig.tight_layout()
    fig.savefig(OUT / "angle_grid.png", dpi=130)

    print(g.label.value_counts().to_dict())
    print(g[(g.elbow_deg.isin([125, 135])) & (g.knee_deg.isin([155, 165]))].to_string(index=False))
    print(occ.to_string(index=False))
