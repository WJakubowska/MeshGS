class Vertex:
    def __init__(self, x, y, z, index = None):
        self.x = x
        self.y = y
        self.z = z
        self.coordinates = (x, y, z)
        self.index = index

    def __str__(self):
      # return f"({int(self.x)}, {int(self.y)}, {int(self.z)})"
        return f"({self.x}, {self.y}, {self.z})"

    def to_dict(self):
        return {'x': self.x, 'y': self.y, 'z': self.z}

    def __lt__(self, other):
        return self.coordinates < other.coordinates

    def __eq__(self, other):
        return isinstance(other, Vertex) and self.coordinates == other.coordinates

    def __hash__(self):
        return hash(self.coordinates)