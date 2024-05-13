from MeshGS.mesh_utils.icosphere import Icosphere
from utils.display_sphere import *
from MeshGS.mesh_utils.triangles_utils import *
from MeshGS.mesh_utils.triangle import Triangle
from torch import nn
import torch 
from utils.gs import setup_training_input
import numpy as np
import pytest
from main import *

center_point = (400, 400, 0)
radius = [285, 140]
n_subdivisions = 2
icosphere = Icosphere(n_subdivisions, center_point, radius)
vertices, triangles = icosphere.vertices, icosphere.triangles
unique_triangles = get_unique_triangles(triangles)
unique_vertices = icosphere.get_all_vertices()
result_triangles = get_triangles_as_indices(unique_vertices, triangles)
xyz, features_dc, features_rest, opacity, vertices_tensor = setup_training_input(unique_vertices, result_triangles)
xyz_tensor = xyz.float().long()
torch.save(xyz_tensor, 'test_data/faces.pt')
torch.save(vertices_tensor, 'test_data/vertices.pt')
loaded_ver = torch.load('test_data/vertices.pt')
loaded_fa = torch.load('test_data/faces.pt')

def test_calculate_barycentric_coordinates():
    point = torch.tensor([[1.0, 2.0, 3.0]])
    vertices_A = torch.tensor([[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [4.0, 5.0, 5.0]])
    vertices_B = torch.tensor([[3.0, 1.0, 3.0], [1.0, 1.0, 3.0], [6.0, 7.0, 5.0]])
    vertices_C = torch.tensor([[2.0, 3.0, 3.0], [1.0, 1.0, 3.0], [8.0, 3.0, 5.0]])

    u, v, w = calculate_barycentric_coordinates(point, vertices_A, vertices_B, vertices_C)
    # print(parameters)
    assert torch.allclose(u, torch.tensor([0.0]))
    assert torch.allclose(v, torch.tensor([0.5]))
    assert torch.allclose(w, torch.tensor([0.5]))

# def test_check_if_outside_point_is_in_triangle():
#     point_outside = torch.tensor([[3.0, 3.0, 3.0]])
#     vertices_A = torch.tensor([[0.0, 0.0, 0.0]])
#     vertices_B = torch.tensor([[2.0, 0.0, 0.0]])
#     vertices_C = torch.tensor([[0.0, 2.0, 0.0]])

#     result_outside = check_if_point_is_in_triangle(point_outside, vertices_A, vertices_B, vertices_C)
#     assert result_outside is False

# def test_check_if_point_is_in_triangle():
#     point = torch.tensor([[1.0, 1.0, 1.0]])
#     vertices_A = torch.tensor([[0.0, 0.0, 0.0]])
#     vertices_B = torch.tensor([[2.0, 0.0, 0.0]])
#     vertices_C = torch.tensor([[0.0, 2.0, 0.0]])

#     result = check_if_point_is_in_triangle(point, vertices_A, vertices_B, vertices_C)
#     assert torch.allclose(result, torch.tensor([[1.0, 1.0, 0.]]))


pytest.main(["-v"])



