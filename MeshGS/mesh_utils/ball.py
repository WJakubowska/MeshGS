from mesh_utils.vertex import Vertex
from mesh_utils.triangle import Triangle
import math

def uv_sphere(n_slices, n_stacks):
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
