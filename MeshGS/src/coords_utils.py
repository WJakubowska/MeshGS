import torch


def calculate_barycentric_coordinates(points, vertices_A, vertices_B, vertices_C):
    # https://ceng2.ktu.edu.tr/~cakir/files/grafikler/Texture_Mapping.pdf
    v0 = vertices_B - vertices_A
    v1 = vertices_C - vertices_A
    v2 = points - vertices_A

    d00 = torch.sum(v0 * v0, dim=1)
    d01 = torch.sum(v0 * v1, dim=1)
    d11 = torch.sum(v1 * v1, dim=1)
    d20 = torch.sum(v2 * v0, dim=1)
    d21 = torch.sum(v2 * v1, dim=1)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u, v, w


def find_uv_coordinates(points, face_indices, faces, vertices):
    faces = faces[face_indices].long()
    A = vertices[faces[:, 0]]
    B = vertices[faces[:, 1]]
    C = vertices[faces[:, 2]]

    points = torch.tensor(points)

    u, v, w = calculate_barycentric_coordinates(points, A, B, C)
    return torch.stack([u, v], dim=1)
