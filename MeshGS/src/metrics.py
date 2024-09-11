from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import structural_similarity_index_measure
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from scipy.spatial import KDTree
import torch.nn.functional as F
import numpy as np
import torch

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))
l1_loss = lambda x, y: F.l1_loss(x, y)
img2BCE = lambda input, target: F.binary_cross_entropy(input, target)
calculate_ssim = lambda preds, target: structural_similarity_index_measure(
    preds.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0)
)

def calculate_lpips(preds, target):
    lpips = LearnedPerceptualImagePatchSimilarity()
    return lpips(preds.permute(2, 0, 1).unsqueeze(0).float(), target.permute(2, 0, 1).unsqueeze(0).float())

def save_metric(text, file):
    with open(file, 'a') as plik:
        plik.write(f"{text}\n\n")

def calculate_chamfer_distance_medium(mesh, mesh_target):
    num_points = 100000
    points = mesh.sample(num_points) 
    points_target = mesh_target.sample(num_points)
    tree = KDTree(points_target) 
    dist_A = tree.query(points)[0] 
    tree = KDTree(points) 
    dist_B = tree.query(points_target)[0]
    return np.mean(dist_A) + np.mean(dist_B)

def calculate_chamfer_distance_loss(mesh, mesh_target):
    num_points = 20000
    points = sample_points_from_meshes(mesh, num_samples=num_points)
    points_target = sample_points_from_meshes(mesh_target, num_samples=num_points)
    loss_chamfer, _ = chamfer_distance(points, points_target)
    return loss_chamfer.item()
    

# def get_normals_at_sampled_points(mesh, sampled_points):
#     tree = cKDTree(mesh.vertices)
#     _, indices = tree.query(sampled_points)
#     return mesh.vertex_normals[indices]

# def normal_consistency(normals1, normals2):
#     dot_product = np.einsum('ij,ij->i', normals1, normals2)
#     return np.mean(dot_product)

# def calculate_normal_consistency(mesh1, mesh2):
#     points1 = mesh1.sample(10000) 
#     points2 = mesh2.sample(10000)

#     normals_sampled1 = get_normals_at_sampled_points(mesh1, points1)
#     normals_sampled2 = get_normals_at_sampled_points(mesh2, points2)
    
#     return normal_consistency(normals_sampled1, normals_sampled2)