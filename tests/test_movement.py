import numpy as np
import pytest

from posture import movement
from posture.keypoints import L_ANKLE, L_ELBOW, L_KNEE, L_WRIST, R_ANKLE, visible
from posture.synthetic import hide, rotate, standing_pose


def test_joint_angle_right_angle_and_straight():
    assert movement.joint_angle((0, 1), (0, 0), (1, 0)) == pytest.approx(90, abs=0.1)
    assert movement.joint_angle((0, 1), (0, 0), (0, -1)) == pytest.approx(180, abs=0.1)


def test_joint_angle_invariant_to_rotation():
    k, _ = standing_pose()
    a = movement.joint_angle(k[5], k[7], k[9])
    kr = rotate(k, 37)
    assert movement.joint_angle(kr[5], kr[7], kr[9]) == pytest.approx(a, abs=1e-3)


def _runner():
    k, c = standing_pose()
    k = k.copy()
    k[L_WRIST] = k[L_ELBOW] + np.array([20.0, -10.0])      # forearm folded up
    k[L_ANKLE] = k[L_KNEE] + np.array([30.0, -20.0])       # shin folded back
    return k, c


def test_bent_arm_and_leg_is_running():
    k, c = _runner()
    assert movement.classify_movement(k, c)[0] == movement.RUNNING


def test_standing_is_non_running():
    k, c = standing_pose()
    assert movement.classify_movement(k, c)[0] == movement.NON_RUNNING


def test_bent_arm_only_is_not_running():
    k, c = standing_pose()
    k = k.copy()
    k[L_WRIST] = k[L_ELBOW] + np.array([20.0, -10.0])
    assert movement.classify_movement(k, c)[0] == movement.NON_RUNNING


def test_one_leg_hidden_gives_unknown():
    k, c = standing_pose()
    label, _ = movement.classify_movement(k, hide(c, [L_ANKLE]))
    assert label == movement.UNKNOWN


def test_hidden_legs_never_give_running():
    k, c = _runner()
    label, _ = movement.classify_movement(k, hide(c, [L_ANKLE, R_ANKLE]))
    assert label == movement.UNKNOWN


def test_thresholds_are_configurable():
    k, c = _runner()
    assert movement.classify_movement(k, c, arm_threshold=1, leg_threshold=1)[0] == movement.NON_RUNNING


def test_visible_helper():
    _, c = standing_pose()
    assert visible(c, [5, 6], 0.3)
    assert not visible(hide(c, [5]), [5, 6], 0.3)
