import torch

class Triangle:
    def __init__(self, v0, v1, v2):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.vertices = [self.v0, self.v1, self.v2]

    def __str__(self):
        return f"v0={self.v0}, v1={self.v1}, v2={self.v2}"

    def to_dict(self):
        return {'v0': self.v0.to_dict(), 'v1': self.v1.to_dict(), 'v2': self.v2.to_dict()}

    def __eq__(self, other):
        if isinstance(other, Triangle):
            return (self.v0 == other.v0 and self.v1 == other.v1 and self.v2 == other.v2) or \
                   (self.v0 == other.v1 and self.v1 == other.v2 and self.v2 == other.v0) or \
                   (self.v0 == other.v2 and self.v1 == other.v0 and self.v2 == other.v1)

    def get_vertices(self):
        return self.vertices

    def get_vertices_tensor(self):
        return [torch.tensor([self.v0.x, self.v0.y, self.v0.z], dtype=torch.float32),
                torch.tensor([self.v1.x, self.v1.y, self.v1.z], dtype=torch.float32),
                torch.tensor([self.v2.x, self.v2.y, self.v2.z], dtype=torch.float32)]