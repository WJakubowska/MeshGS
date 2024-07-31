import torch
import trimesh
import numpy as np
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils.triangles_utils import get_triangles_as_indices
from utils.icosphere import Icosphere
from utils.ball import *
from utils.vertex import Vertex
from utils.display_sphere import *
from collections import defaultdict


# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))
calculate_ssim = lambda preds, target: structural_similarity_index_measure(
    preds.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0)
)
l1_loss = lambda x, y: F.l1_loss(x, y)
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
img2BCE = lambda input, target: F.binary_cross_entropy(input, target)

def calculate_lpips(preds, target):
    lpips = LearnedPerceptualImagePatchSimilarity()
    return lpips(preds.permute(2, 0, 1).unsqueeze(0).float(), target.permute(2, 0, 1).unsqueeze(0).float())

# Model
class MeshGS(torch.nn.Module):
    def __init__(self, mesh_path=None, n_subdivisions=0, train_vertices=True, texture_size=8):
        super(MeshGS, self).__init__()

        self.opacity = None
        self.vertices = None
        self.faces = None
        self.texture = None
        self.offsets = None
        self.background = None
        self.mesh_path = mesh_path
        self.train_vertices = train_vertices
        self.texture_size = texture_size

        if mesh_path is not None:
            mesh = trimesh.load(mesh_path, force="mesh")

            centroid = mesh.centroid
            mesh.vertices -= centroid

            if n_subdivisions != 0:
                mesh = self.densify_mesh(mesh, n_subdivisions)

            vertices = []
            for vertex in mesh.vertices:
                vertex = Vertex(vertex[0], vertex[1], vertex[2])
                vertices.append(vertex)

            self.mesh_vertices = vertices
            self.mesh_faces = mesh.faces

        else:
            raise ValueError("Path to mesh file is not provided")

        self.setup_training_input(self.mesh_vertices, self.mesh_faces)

    def create_mesh_icosphere(
        self, center_point=(0, 0, 0), radius=[1], n_subdivisions=2
    ):
        mesh = Icosphere(n_subdivisions, center_point, radius)
        vertices, triangles = mesh.vertices, mesh.triangles
        unique_vertices = mesh.get_all_vertices()

        triangles = get_triangles_as_indices(unique_vertices, triangles)
        return unique_vertices, triangles

    def create_mesh_sphere(self, n_slices=18, n_stacks=18):
        unique_vertices, triangles = uv_sphere(n_slices, n_stacks)
        plot_sphere(unique_vertices, triangles, show_vertices=False)
        triangles = get_triangles_as_indices(unique_vertices, triangles)
        return unique_vertices, triangles

    def get_vertices(self):
        return self.vertices

    def get_faces(self):
        return self.faces

    def get_opacity(self):
        return self.opacity

    def get_texture(self):
        return self.texture

    def get_offsets(self):
        return self.offsets

    def get_background(self):
        return self.background

    def inverse_sigmoid(self, x):
        return torch.log(x / (1 - x))

    def densify_mesh(self, mesh, subdivisions=1):
        for _ in range(subdivisions):
            mesh = mesh.subdivide()
        return mesh

    def setup_training_input(self, mesh_vertices, mesh_faces):

        vertices_list = [(vertex.x, vertex.y, vertex.z) for vertex in mesh_vertices]
        vertices = torch.tensor(vertices_list, dtype=torch.float64, requires_grad=False)
        faces = torch.tensor(mesh_faces, dtype=torch.float64, requires_grad=False)
        num_faces = faces.shape[0]
        num_vertices = vertices.shape[0]

        self.faces = torch.nn.Parameter(faces, requires_grad=False)
        self.vertices = torch.nn.Parameter(vertices, requires_grad=False)

        if self.train_vertices == True:
            self.offsets = torch.nn.Parameter(torch.zeros(num_vertices, 3), requires_grad=True)
        else:
            self.offsets = torch.nn.Parameter(torch.zeros(num_vertices, 3), requires_grad=False)

        self.background = torch.nn.Parameter(torch.zeros(num_vertices), requires_grad=False)

        self.opacity = torch.nn.Parameter(self.inverse_sigmoid(0.1 * torch.ones(num_faces)),requires_grad=True)
        self.texture = torch.nn.Parameter(torch.randn(num_faces, self.texture_size, self.texture_size, 3) * 0.01, requires_grad=True)

        print("Mesh path: ", self.mesh_path)
        print("Vertices shape: ", self.vertices.shape)
        print("Offset shape: ", self.offsets.shape)
        print("Triangles shape: ", self.faces.shape)
        print("Opacity shape: ", self.opacity.shape)
        print("Texture shape: ", self.texture.shape)


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing="ij"
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1
    )
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    dirs = np.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1
    )
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = (
        -1.0
        / (W / (2.0 * focal))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = (
        -1.0
        / (H / (2.0 * focal))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Ray-tracking helpers


def sort_faces_along_the_ray_by_distance(points, index_ray, index_tri, rays_o):
    intersections_by_ray = defaultdict(list)

    for point, ray_idx, tri_idx in zip(points, index_ray, index_tri):
        distance = torch.linalg.norm(torch.tensor(point) - rays_o[ray_idx])
        intersections_by_ray[ray_idx].append((distance, tri_idx, point))

    for ray_idx in intersections_by_ray:
        intersections_by_ray[ray_idx].sort(key=lambda x: x[0])

    return intersections_by_ray


def map_ray_to_intersections(intersections_by_ray, N_rays):
    intersects_faces_dict = defaultdict(list)  # {ray_idx}: [face_idx]
    intersects_points_dict = defaultdict(list)

    for ray_index in intersections_by_ray:
        sorted_face_indices = [
            face_idx for distance, face_idx, point in intersections_by_ray[ray_index]
        ]
        sorted_point = [
            point for distance, face_idx, point in intersections_by_ray[ray_index]
        ]
        intersects_faces_dict[ray_index] = sorted_face_indices
        intersects_points_dict[ray_index] = sorted_point

    intersects_faces_dict = dict(intersects_faces_dict)
    additional_keys = {
        key: None for key in range(N_rays) if key not in intersects_faces_dict
    }
    intersects_faces_dict.update(additional_keys)
    intersects_faces_dict = dict(sorted(intersects_faces_dict.items()))

    intersects_points_dict = dict(intersects_points_dict)
    intersects_points_dict.update(additional_keys)
    intersects_points_dict = dict(sorted(intersects_points_dict.items()))

    return intersects_faces_dict, intersects_points_dict


# UV coordinates helpers


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
