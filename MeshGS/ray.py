from ray_tracing.rays import Rays
from ray_tracing.mesh import Mesh
import torch
import numpy as np


def find_intersection_points_with_mesh(vertices, faces, rays_o, rays_d, plot: bool = True):

    rays = Rays(
        origin=rays_o,
        direction=rays_d,
    )

    # vertices = torch.load('vertices.pt').double()
    # faces = torch.load('faces.pt')

    mesh = Mesh(
        vertices=vertices,
        faces=faces
    )

    out = rays.find_intersection_points_with_mesh(
        mesh=mesh,
        plot=plot
    )

    return out