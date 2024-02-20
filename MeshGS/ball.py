from vertex import Vertex
from triangle import Triangle
import torch

def create_sphere_mesh(radius, num_points):
    vertices = []

    for phi in torch.linspace(0, torch.pi, num_points):
        for theta in torch.linspace(0, 2 * torch.pi, num_points):
            x = radius * torch.sin(phi) * torch.cos(theta)
            y = radius * torch.sin(phi) * torch.sin(theta)
            z = radius * torch.cos(phi)

            vertex = Vertex(x, y, z)
            vertices.append(vertex)

    triangles = []

    for i in range(num_points - 1):
        for j in range(num_points - 1):
            v0 = vertices[i * num_points + j]
            v1 = vertices[i * num_points + j + 1]
            v2 = vertices[(i + 1) * num_points + j]
            v3 = vertices[(i + 1) * num_points + j + 1]

            triangles.append(Triangle(v0, v1, v2))
            triangles.append(Triangle(v1, v3, v2))

    return vertices, triangles