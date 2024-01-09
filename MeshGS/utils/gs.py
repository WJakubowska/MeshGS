from colors import SH2RGB, RGB2SH
import numpy as np
import torch
from torch import nn

C0 = 0.28209479177387814

def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def setup_training_input(unique_vertices, result_triangles):
    num_pts = 100_000
    max_sh_degree = 3
    print(f"Generating random point cloud ({num_pts})...")

    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    points = xyz
    shs = np.random.random((num_pts, 3)) / 255.0
    colors=SH2RGB(shs)

    # fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
    # fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())
    # features = torch.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2)).float().cuda()
    fused_point_cloud = torch.tensor(np.asarray(points)).float()
    fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float())
    features = torch.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2)).float()
    features[:, :3, 0 ] = fused_color
    features[:, 3:, 1:] = 0.0

    # opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
    opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cpu"))

    coordinates_list = [(vertex.x, vertex.y, vertex.z) for vertex in unique_vertices]
    vertices_tensor = torch.tensor(coordinates_list, dtype=torch.float64, requires_grad=True)
  
    vertices_table = nn.Parameter(vertices_tensor)

    r_triangles = torch.tensor(result_triangles, dtype=torch.float64, requires_grad=False)
    xyz = nn.Parameter(r_triangles.requires_grad_(False))  # odpowiednik to wierzchołki grafu
    features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
    features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
    opacity = nn.Parameter(opacities.requires_grad_(True))
    print("Shape:")
    print("vertices_table: ", vertices_table.shape)
    print("xyz: ", xyz.shape)
    print("features_dc: ", features_dc.shape)
    print("features_rest: ", features_rest.shape)
    print("opacity: ", opacity.shape)
    return xyz, features_dc, features_rest, opacity, vertices_tensor