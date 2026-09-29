import numpy as np


class vct3:
    """A 3-D column vector, expressed in millimetres."""

    def __init__(self, x, y=None, z=None):
        if y is None and z is None:
            values = np.asarray(x, dtype=float).reshape(3)
        elif y is not None and z is not None:
            values = np.array([x, y, z], dtype=float)
        else:
            raise TypeError("vct3 requires either one length-3 value or x, y, z")
        self.vec = values.reshape(3, 1)

    def norm(self):
        return float(np.linalg.norm(self.vec))

    def __add__(self, other):
        return vct3(self.vec + _as_vct3(other).vec)

    def __sub__(self, other):
        return vct3(self.vec - _as_vct3(other).vec)

    def __neg__(self):
        return vct3(-self.vec)

    def __mul__(self, scalar):
        return vct3(self.vec * scalar)

    __rmul__ = __mul__

    def __repr__(self):
        x, y, z = self.vec.ravel()
        return f"vct3({x:.6g}, {y:.6g}, {z:.6g})"


def _as_vct3(value):
    return value.p if hasattr(value, "p") else value


class vct3Array:
    def __init__(self, points):
        self.mat = np.column_stack([_as_vct3(point).vec for point in points])

    def mean(self):
        return vct3(np.mean(self.mat, axis=1))

    def __sub__(self, point):
        return vct3Array([vct3(self.mat[:, index] - _as_vct3(point).vec.ravel())
                          for index in range(self.mat.shape[1])])


class Rot:
    def __init__(self, matrix=None):
        self.matrix = np.eye(3) if matrix is None else np.asarray(matrix, dtype=float).reshape(3, 3)

    @classmethod
    def xyz(cls, rx, ry, rz):
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        rx_mat = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry_mat = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz_mat = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return cls(rz_mat @ ry_mat @ rx_mat)

    def __repr__(self):
        return f"Rot(matrix={self.matrix!r})"


class Frame:
    """Rigid transform with the convention ``F * point``."""

    def __init__(self, rotation=None, position=None):
        if position is None and rotation is not None and np.asarray(rotation).shape == (4, 4):
            transform = np.asarray(rotation, dtype=float)
            self.R = Rot(transform[:3, :3])
            self.p = vct3(transform[:3, 3])
        else:
            self.R = rotation if isinstance(rotation, Rot) else Rot(rotation)
            self.p = vct3([0, 0, 0]) if position is None else _as_vct3(position)

    @classmethod
    def eye(cls):
        return cls(Rot(), vct3(0, 0, 0))

    @property
    def matrix(self):
        transform = np.eye(4)
        transform[:3, :3] = self.R.matrix
        transform[:3, 3] = self.p.vec.ravel()
        return transform

    def inv(self):
        r_inv = self.R.matrix.T
        return Frame(Rot(r_inv), vct3(-r_inv @ self.p.vec))

    def __mul__(self, other):
        if isinstance(other, Frame):
            return Frame(Rot(self.R.matrix @ other.R.matrix),
                         vct3(self.R.matrix @ other.p.vec + self.p.vec))
        point = _as_vct3(other)
        result = vct3(self.R.matrix @ point.vec + self.p.vec)
        if hasattr(other, "p"):
            from .uncertain_types import uvct3
            return uvct3(result)
        return result

    def __repr__(self):
        return f"Frame({self.R!r}, {self.p!r})"
