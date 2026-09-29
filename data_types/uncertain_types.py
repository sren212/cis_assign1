from .nominal_types import Frame, Rot, vct3


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

    def __mul__(self, other):
        return self.F * other

    def __repr__(self):
        return f"uFrame({self.F!r})"
