class Vertex:
    def __init__(self, x, y, z, index=None):
        self.x = x
        self.y = y
        self.z = z
        self.index = index

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def to_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z}

    def __lt__(self, other):
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)

    def __eq__(self, other):
        return isinstance(other, Vertex) and (self.x, self.y, self.z) == (
            other.x,
            other.y,
            other.z,
        )

    def __sub__(self, other):
        return

    def __hash__(self):
        return hash((self.x, self.y, self.z))
