import numpy as np


def normalize_angle_0_to_2pi(angle):
    return angle % (2 * np.pi)
