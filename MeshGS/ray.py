import torch
import numpy as np
import plotly.graph_objects as go

import plotly.io as pio
from pathlib import Path


class Rays:
    def __init__(
            self,
            origin: torch.Tensor,
            direction: torch.Tensor
    ):
        self.origin = origin
        self.direction = direction

    def _dot_product(self, Y, X, P, n_expanded):
        cross_product = torch.cross(Y - X, P - X, dim=-1)
        dot_product = (cross_product * n_expanded).sum(dim=-1)
        return dot_product

    def find_intersection_points_with_mesh(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor
    ):

        triangle_vertices = vertices[faces]

        num_rays = self.origin.shape[0]
        num_triangles = triangle_vertices.shape[0]

        # Triangle
        A = triangle_vertices[:, 0]
        B = triangle_vertices[:, 1]
        C = triangle_vertices[:, 2]

        AB = B - A  # Oriented segment A to B
        AC = C - A  # Oriented segment A to C
        n = torch.cross(AB, AC)  # Normal vector
        n_ = n / torch.linalg.norm(n, dim=1, keepdim=True)  # Normalized normal

        # expand
        n_expanded = n_.unsqueeze(0).expand(num_rays, num_triangles, 3)

        ray_origins_expanded = self.origin.unsqueeze(1).expand(num_rays, num_triangles, 3)

        ray_directions_norm = self.direction / torch.linalg.norm(
            self.direction, dim=1, keepdim=True
        )  # Unit vector (versor) of e => ê

        ray_directions_norm_expanded = ray_directions_norm.unsqueeze(1).expand(
            num_rays, num_triangles, 3
        )

        A_expand = A.unsqueeze(0).expand(num_rays, num_triangles, 3)
        B_expand = B.unsqueeze(0).expand(num_rays, num_triangles, 3)
        C_expand = C.unsqueeze(0).expand(num_rays, num_triangles, 3)

        # Using the point A to find d
        d = -(n_expanded * A_expand).sum(dim=-1)

        # Finding parameter t
        t = -((n_expanded * ray_origins_expanded).sum(dim=-1) + d)
        tt = (n_expanded * ray_directions_norm_expanded).sum(dim=-1)
        t /= tt

        # Finding P [num_rays, num_triangles, 3D point]
        pts = ray_origins_expanded + t.unsqueeze(-1) * ray_directions_norm_expanded

        # Get the resulting vector for each vertex
        # following the construction order
        Pa = self._dot_product(B_expand, A_expand, pts, n_expanded)
        Pb = self._dot_product(C_expand, B_expand, pts, n_expanded)
        Pc = self._dot_product(A_expand, C_expand, pts, n_expanded)

        backface_intersection = torch.where(t < 0, 0, 1)

        valid_point = (Pa > 0) & (Pb > 0) & (Pc > 0)  # [num_rays, num_triangles]

        _d = pts - ray_origins_expanded
        _d = (_d ** 2).sum(dim=2)

        d_valid = valid_point.int() * _d
        d_valid_inv = - torch.log(d_valid.abs())

        idx = d_valid_inv.abs().min(dim=1).indices
        nearest_valid_point_mask = torch.zeros_like(d_valid_inv)
        nearest_valid_point_mask[torch.arange(num_rays), idx] = 1
        nearest_valid_point_mask = (d_valid_inv != 0) * nearest_valid_point_mask

        idxs = torch.where(nearest_valid_point_mask == 1)
        pts_nearest = pts[idxs]

        nearest_points = nearest_valid_point_mask * valid_point
        nearest_points_idx = torch.where(nearest_points == 1)
        pts_nearest_each_ray = torch.zeros(num_rays, 3).double()
        pts_nearest_each_ray[nearest_points_idx[0].long()] = pts[nearest_points_idx].double()

        out = {
            'pts': pts,
            'backface_intersection': backface_intersection,
            'valid_point': valid_point,
            'nearest_valid_point_mask': nearest_valid_point_mask,
            'pts_nearest': pts_nearest,
            'nearest_points_idx': nearest_points_idx,
            'pts_nearest_each_ray': pts_nearest_each_ray,
            'd': _d,
            'Pa': Pa,
            'Pb': Pb,
            'Pc': Pc
        }
        
        return out

def find_intersection_points_with_mesh(vertices, faces, rays_o, rays_d):

    rays = Rays(
        origin=rays_o,
        direction=rays_d,
    )

    out = rays.find_intersection_points_with_mesh(
        vertices=vertices,
        faces=faces.long()
    )

    return out