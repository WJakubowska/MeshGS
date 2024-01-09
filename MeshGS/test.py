from icosphere import Icosphere
from utils.display_sphere import plot_sphere, plot_filled_triangle_sphere, plot_sphere_from_tensor_with_index, plot_rays_mesh_and_points
from utils.triangles_utils import get_unique_triangles, get_triangles_as_indices
from triangle import Triangle
from torch import nn
import torch 
from utils.gs import setup_training_input
from ray import find_intersection_points_with_mesh, get_rays, get_rays_np
import cv2
import numpy as np

center_point = (400, 400, 0)
radius = [285, 140]
n_subdivisions = 2
icosphere = Icosphere(n_subdivisions, center_point, radius)
vertices, triangles = icosphere.vertices, icosphere.triangles
# plot_sphere(vertices, triangles)
unique_triangles = get_unique_triangles(triangles)
# plot_filled_triangle_sphere(vertices, triangles, fill_triangle_index=5, show_vertices=False)
# print(len(unique_triangles))

uv = icosphere.get_unique_vertices()
# print("len get_unique_vertices():", len(uv))

unique_vertices = icosphere.get_all_vertices()
# print("len get_all_vertices(): ", len(unique_vertices))

result_triangles = get_triangles_as_indices(unique_vertices, triangles)

xyz, features_dc, features_rest, opacity, vertices_tensor = setup_training_input(unique_vertices, result_triangles)

xyz_tensor = xyz.float().long()
torch.save(xyz_tensor, 'faces.pt')
torch.save(vertices_tensor, 'vertices.pt')
loaded_ver = torch.load('vertices.pt')
loaded_fa = torch.load('faces.pt')

ver = loaded_ver.tolist()
tri = loaded_fa.tolist()

# plot_sphere_from_tensor_with_index(ver, tri)
# find_intersection_points_with_mesh()

transform_matrix = [[
                    -1.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    -0.41173306107521057,
                    0.9113044738769531,
                    3.673585891723633
                ],
                [
                    0.0,
                    0.9113044142723083,
                    0.41173309087753296,
                    1.659749150276184
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]]

image_path = 'C:/Users/48783/Desktop/GS+NeRF+Mesh/MeshGS/r_0.png'
image = cv2.imread(image_path)
H, W, _ = image.shape
camera_angle_x = 0.6911112070083618
focal = .5 * W / np.tan(.5 * camera_angle_x)
# c2w = np.array(transform_matrix)[:3, :4]
c2w = np.array(transform_matrix)
K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

c2w = torch.tensor(c2w)
rays_o, rays_d = get_rays(H, W, K, c2w)
rays_o = rays_o.reshape((-1, 3))
rays_d = rays_d.reshape((-1, 3))
selected_indices = np.random.choice(len(rays_o), 500, replace=False)
rays_o = rays_o[selected_indices]
rays_d = rays_d[selected_indices]

print(rays_o.shape)
torch.save(rays_o, 'rays_origins.pt')
torch.save(rays_d, 'rays_directions.pt')
loaded_o = torch.load('rays_origins.pt')
# print(loaded_o)
out = find_intersection_points_with_mesh(False)
points=out['pts'][out['valid_point']]
print("p", points.shape)
print(points)
# plot_rays_mesh_and_points(
#     rays_o,
#     rays_d,
#     loaded_ver,
#     loaded_fa,
#     points,
#     500.0,
# )
