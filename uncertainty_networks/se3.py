import numpy as np


def _skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(rotation_vector):
    """Return Exp(rotation_vector) as a 3x3 rotation matrix."""
    rotation_vector = np.asarray(rotation_vector, dtype=float).reshape(3)
    theta = np.linalg.norm(rotation_vector)
    omega_hat = _skew(rotation_vector)
    if theta < 1e-12:
        return np.eye(3) + omega_hat
    return (np.eye(3) + np.sin(theta) / theta * omega_hat
            + (1.0 - np.cos(theta)) / theta**2 * (omega_hat @ omega_hat))


def log_so3(rotation):
    """Return the rotation vector whose exponential is rotation."""
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    cos_theta = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    vee = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    if theta < 1e-12:
        return 0.5 * vee
    return theta / (2.0 * np.sin(theta)) * vee


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
