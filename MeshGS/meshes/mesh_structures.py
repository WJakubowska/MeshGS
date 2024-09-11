import numpy as np
import torch
import math

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
        return [self.v0, self.v1, self.v2]

    def get_vertices_tensor(self):
        return [torch.tensor([self.v0.x, self.v0.y, self.v0.z], dtype=torch.float32),
                torch.tensor([self.v1.x, self.v1.y, self.v1.z], dtype=torch.float32),
                torch.tensor([self.v2.x, self.v2.y, self.v2.z], dtype=torch.float32)]


class Icosphere:
    def __init__(self, n_subdivisions, center_point=(0, 0, 0), radius=[1]):
        self.vertices, self.triangles = self._create_icosphere(center_point, radius, n_subdivisions)

    def _project_to_unit_sphere(self, vertices):
        updated_vertices = []
        for vertex in vertices:
            p = np.array([vertex.x, vertex.y, vertex.z])
            n = np.linalg.norm(p)
            updated_vertex = Vertex(p[0] / n, p[1] / n, p[2] / n)
            updated_vertices.append(updated_vertex)
        return updated_vertices

    def _icosahedron(self, center_point, radius):
        vertices = []
        triangles = []

        phi = (1.0 + np.sqrt(5.0)) * 0.5
        a = 1.0
        b = 1.0 / phi

        v1 = Vertex(0, b, -a)
        v2 = Vertex(b, a, 0)
        v3 = Vertex(-b, a, 0)
        v4 = Vertex(0, b, a)
        v5 = Vertex(0, -b, a)
        v6 = Vertex(-a, 0, b)
        v7 = Vertex(0, -b, -a)
        v8 = Vertex(a, 0, -b)
        v9 = Vertex(a, 0, b)
        v10 = Vertex(-a, 0, -b)
        v11 = Vertex(b, -a, 0)
        v12 = Vertex(-b, -a, 0)

        vertex_tab = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12]

        for v in vertex_tab:
            v.x = center_point[0] + radius * v.x
            v.y = center_point[1] + radius * v.y
            v.z = center_point[2] + radius * v.z

        vertices.extend(vertex_tab)

        vertices = self._project_to_unit_sphere(vertices)

        triangles.extend([Triangle(v3, v2, v1), Triangle(v2, v3, v4),
                          Triangle(v6, v5, v4), Triangle(v5, v9, v4),
                          Triangle(v8, v7, v1), Triangle(v7, v10, v1),
                          Triangle(v12, v11, v5), Triangle(v11, v12, v7),
                          Triangle(v10, v6, v3), Triangle(v6, v10, v12),
                          Triangle(v9, v8, v2), Triangle(v8, v9, v11),
                          Triangle(v3, v6, v4), Triangle(v9, v2, v4),
                          Triangle(v10, v3, v1), Triangle(v2, v8, v1),
                          Triangle(v12, v10, v7), Triangle(v8, v11, v7),
                          Triangle(v6, v12, v5), Triangle(v11, v9, v5)])

        return vertices, triangles

    def _loop_subdivision(self, vertices, triangles):
        new_vertices = list(vertices)
        new_triangles = []

        for triangle in triangles:
            v0, v1, v2 = triangle.v0, triangle.v1, triangle.v2

            v01 = Vertex((v0.x + v1.x) / 2, (v0.y + v1.y) / 2, (v0.z + v1.z) / 2)
            v12 = Vertex((v1.x + v2.x) / 2, (v1.y + v2.y) / 2, (v1.z + v2.z) / 2)
            v20 = Vertex((v2.x + v0.x) / 2, (v2.y + v0.y) / 2, (v2.z + v0.z) / 2)

            new_vertices.extend([v01, v12, v20])

            v01_idx = len(new_vertices) - 3
            v12_idx = len(new_vertices) - 2
            v20_idx = len(new_vertices) - 1

            new_triangles.extend([Triangle(v0, v01, v20),
                                  Triangle(v1, v12, v01),
                                  Triangle(v2, v20, v12),
                                  Triangle(new_vertices[v01_idx], new_vertices[v12_idx], new_vertices[v20_idx])])

        return new_vertices, new_triangles
    
    def _create_icosphere(self, center_point, radius_tab, n_subdivisions):
        triangles_tab = []
        vertices_tab = []

        for radius in radius_tab:
            vertices, triangles = self._icosahedron(center_point, radius)
            for _ in range(n_subdivisions):
                vertices, triangles = self._loop_subdivision(vertices, triangles)
                vertices = self._project_to_unit_sphere(vertices.copy())
            triangles_tab.append(triangles)
            vertices_tab.append(vertices)


        vertices_tab = [vertex for vertices in vertices_tab for vertex in vertices]
        triangles_tab = [triangle for triangles in triangles_tab for triangle in triangles]

        return vertices_tab, triangles_tab

    def count_unique_vertices(self):
        unique_vertices_set = set(self.vertices)
        count_unique = len(unique_vertices_set)
        return count_unique

    def get_unique_vertices(self):
        unique_vertices_set = set(self.vertices)
        return list(unique_vertices_set)

    def get_all_vertices(self):
        all_vertices_set = set()
        for triangle in self.triangles:
            all_vertices_set.update(triangle.get_vertices())
        return list(all_vertices_set)
    

class UVSphere:
    def __init__(self, n_slices=10, n_stacks=10):
        self.vertices, self.triangles = self._create_sphere(n_slices, n_stacks)

    
    def __uv_sphere(self, n_slices, n_stacks):
        vertices = []
        vertices.append(Vertex(0, 1, 0))
        for i in range(n_stacks - 1):
            phi = math.pi * (i + 1) / n_stacks
            for j in range(n_slices):
                theta = 2.0 * math.pi * j / n_slices
                x = math.sin(phi) * math.cos(theta)
                y = math.cos(phi)
                z = math.sin(phi) * math.sin(theta)
                vertices.append(Vertex(x, y, z))
        vertices.append(Vertex(0, -1, 0))

        triangles = []

        for i in range(n_slices):
            i0 = i + 1
            i1 = (i + 1) % n_slices + 1
            triangles.append(Triangle(vertices[0], vertices[i1], vertices[i0]))
            i0 = i + n_slices * (n_stacks - 2) + 1
            i1 = (i + 1) % n_slices + n_slices * (n_stacks - 2) + 1
            triangles.append(Triangle(vertices[-1], vertices[i0], vertices[i1]))

        for j in range(n_stacks - 2):
            j0 = j * n_slices + 1
            j1 = (j + 1) * n_slices + 1
            for i in range(n_slices):
                i0 = j0 + i
                i1 = j0 + (i + 1) % n_slices
                i2 = j1 + (i + 1) % n_slices
                i3 = j1 + i
                triangles.append(Triangle(vertices[i0], vertices[i1], vertices[i2]))
                triangles.append(Triangle(vertices[i0], vertices[i2], vertices[i3]))

        return vertices, triangles