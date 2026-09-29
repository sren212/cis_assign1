import numpy as np


def _skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_se3(twist):
    """Return Exp([rotation-vector, translation]) as a 4x4 transform."""
    twist = np.asarray(twist, dtype=float).reshape(6)
    omega, translation = twist[:3], twist[3:]
    theta = np.linalg.norm(omega)
    omega_hat = _skew(omega)
    if theta < 1e-12:
        rotation = np.eye(3) + omega_hat
        jacobian = np.eye(3) + 0.5 * omega_hat
    else:
        rotation = (np.eye(3) + np.sin(theta) / theta * omega_hat
                    + (1 - np.cos(theta)) / theta**2 * (omega_hat @ omega_hat))
        jacobian = (np.eye(3) + (1 - np.cos(theta)) / theta**2 * omega_hat
                    + (theta - np.sin(theta)) / theta**3 * (omega_hat @ omega_hat))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = jacobian @ translation
    return transform
