# from ray_tracing.rays import Rays
# from ray_tracing.mesh import Mesh
import torch
import numpy as np
import plotly.graph_objects as go

import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

from torch.optim import Adam

class Rays:
    """Lines in 3D space, each has an origin point and a direction vector."""

    def __init__(
            self,
            origin: torch.Tensor,
            direction: torch.Tensor
    ):
        """Initialize lines with origin points and direction vectors.

        Each line has a corresponding origin point and a direction vector.
        Args:
            origin: torch.Tensor with shape (N, 3),
             where N is the number of lines, and 3 corresponds to
             a set of three coordinates defining a point in 3D space.
            direction: torch.Tensor with shape (N, 3).

        """
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
        print(f'Vertices requires_grad in ray: {vertices.requires_grad}')
        print(f'Faces requires_grad in ray: {faces.requires_grad}')

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

    print(f'Vertices requires_grad in find: {vertices.requires_grad}')
    print(f'Faces requires_grad in find: {faces.requires_grad}')
    

    out = rays.find_intersection_points_with_mesh(
        vertices=vertices,
        faces=faces.long()
    )

    print(f'Vertices requires_grad in find 2: {vertices.requires_grad}')
    print(f'Faces requires_grad in find 2: {faces.requires_grad}')
    

    return out


# vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
# faces = torch.tensor([[0, 1, 2]], requires_grad=False)
# target_point = torch.tensor([0.5, 0.5, 0.0] , requires_grad=True) 
# target_face = faces[0]

# num_rays = 100
# ray_origins = torch.rand(num_rays, 3 , requires_grad=False)
# ray_directions = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=False).repeat(num_rays, 1)

# rays = Rays(ray_origins, ray_directions)
# optimizer = Adam([target_point], lr=0.1)

# for i in range(100):
#     optimizer.zero_grad()
#     pts = rays.find_intersection_points_with_mesh(vertices, faces)
#     distances = torch.norm(pts['pts_nearest_each_ray'] - target_point, dim=1)
#     loss = torch.mean(distances)
#     loss.backward()
#     optimizer.step() 
#     print(f"Iteration {i+1}, Loss: {loss.item()}")
# print("Optimal point:", target_point)


# class RayTracer:
#     def __init__(self, rays_o, rays_d):
#         self.rays_o = rays_o
#         self.rays_d = rays_d

#     def find_intersection_points_with_mesh(self, vertices, faces):
#         faces = faces.long()
    
#         A = vertices[faces[:, 0]] 
#         B = vertices[faces[:, 1]] 
#         C = vertices[faces[:, 2]]  


#         AB = B - A
#         AC = C - A
#         face_normals = torch.cross(AB, AC)

  
#         denom = torch.sum(face_normals.unsqueeze(0) * self.rays_d.unsqueeze(1), dim=-1)


#         valid_ray_mask = torch.abs(denom) > 1e-6

#         AO = A.unsqueeze(0) - self.rays_o.unsqueeze(1)
#         t = torch.sum(AO.unsqueeze(2) * face_normals.unsqueeze(0), dim=-1) / denom.unsqueeze(1)
#         t[~valid_ray_mask] = float('inf')
#         min_t, min_idx = torch.min(t, dim=1)
#         intersection_points = self.rays_o.unsqueeze(1) + min_t.unsqueeze(2) * self.rays_d.unsqueeze(1)


#         fig = go.Figure(data=[
#             go.Scatter3d(
#                 x=intersection_points.detach().numpy()[:, :, 0].flatten(),
#                 y=intersection_points.detach().numpy()[:, :, 1].flatten(),
#                 z=intersection_points.detach().numpy()[:, :, 2].flatten(),
#                 mode='markers',
#                 marker=dict(
#                     size=5,
#                     color='blue',
#                     opacity=0.8
#                 )
#             )
#         ])
#         fig.update_layout(scene=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z'
#         ))
#         dir = 'images'
#         Path(dir).mkdir(parents=True, exist_ok=True)
#         name = "points.html"
#         fig.write_html(
#             f'{dir}/{name}', auto_open=True
#         )


#         print("SHAAAAAAAAPE: ", intersection_points.shape)
#         print(intersection_points)
#         return intersection_points

