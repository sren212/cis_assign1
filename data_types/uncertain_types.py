from .nominal_types import Frame, Rot, vct3
from uncertainty_networks.se3 import exp_se3


class uvct3:
    """A vector paired with optional uncertainty metadata; ``p`` is nominal."""

    def __init__(self, point, cov=None):
        self.p = point.p if isinstance(point, uvct3) else (point if isinstance(point, vct3) else vct3(point))
        self.cov = cov

    def __repr__(self):
        return f"uvct3({self.p!r})"


class uRot:
    def __init__(self, rotation, cov=None):
        self.R = rotation.R if isinstance(rotation, uRot) else (rotation if isinstance(rotation, Rot) else Rot(rotation))
        self.cov = cov


class uFrame:
    def __init__(self, frame, cov=None):
        self.F = frame.F if isinstance(frame, uFrame) else (frame if isinstance(frame, Frame) else Frame(frame))
        self.cov = cov
        self._left_convention = False

    def as_left_convention(self):
        out = uFrame(self)
        out._left_convention = True
        return out

    def perturb(self, twist):
        delta = Frame(exp_se3(twist))
        return delta * self.F if self._left_convention else self.F * delta

    def __mul__(self, other):
        return self.F * other

    def __repr__(self):
        return f"uFrame({self.F!r})"
