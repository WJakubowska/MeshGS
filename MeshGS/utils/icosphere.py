import numpy as np
from utils.vertex import Vertex
from utils.triangle import Triangle

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


