import torch
import trimesh
import numpy as np
import torch.nn.functional as F
from mesh_utils.triangles_utils import  get_triangles_as_indices
from mesh_utils.icosphere import Icosphere
from mesh_utils.ball import *
from mesh_utils.vertex import Vertex



# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
img2BCE = lambda input, target: F.binary_cross_entropy(input, target)


# Model
class MeshGS(torch.nn.Module):
    def __init__(self, mesh_from_file = False, mesh_path = None):
        super(MeshGS, self).__init__()

        self.opacity = None
        self.vertices = None
        self.faces = None
        self.texture = None

        self.mesh_path = mesh_path
        self.test_mesh = False


        if mesh_from_file == True:
            if mesh_path is not None:
                mesh = trimesh.load(mesh_path, force='mesh') 
                vertices = []
                for vertex in mesh.vertices:
                    vertex = Vertex(vertex[0], vertex[1], vertex[2])
                    vertices.append(vertex)

                self.mesh_vertices = vertices
                self.mesh_faces = mesh.faces
              
            else:
                raise ValueError("Path to mesh file is not provided")
            
        self.setup_training_input(self.mesh_vertices, self.mesh_faces)

    def create_mesh_icosphere(self, center_point = (0, 0, 0), radius = [3], n_subdivisions = 2):
        mesh = Icosphere(n_subdivisions, center_point, radius)
        vertices, triangles = mesh.vertices, mesh.triangles
        unique_vertices = mesh.get_all_vertices()
        triangles = get_triangles_as_indices(unique_vertices, triangles)
        return unique_vertices, triangles
    
    def create_mesh_sphere(self, n_slices = 22, n_stacks = 18):
        unique_vertices, triangles = uv_sphere(n_slices, n_stacks)
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
    

    def inverse_sigmoid(self, x):
        return torch.log(x/(1-x))

    def setup_training_input(self, mesh_vertices, mesh_faces):
        
        coordinates_list = [(vertex.x, vertex.y, vertex.z) for vertex in mesh_vertices]
        vertices_tensor = torch.tensor(coordinates_list, dtype=torch.float64, requires_grad=False) 
        r_triangles = torch.tensor(mesh_faces, dtype=torch.float64, requires_grad=False)

        
        self.faces = torch.nn.Parameter(r_triangles, requires_grad=False)
        self.vertices = torch.nn.Parameter(vertices_tensor, requires_grad=False)       
        
     
        self.opacity = torch.nn.Parameter(self.inverse_sigmoid(0.1 * torch.ones(r_triangles.shape[0]+1)), requires_grad=True)
        # self.rgb_color = torch.nn.Parameter(torch.abs(torch.normal(0, 0.1, size=(r_triangles.shape[0]+1, 3))), requires_grad=True)
        self.texture = torch.nn.Parameter(torch.randn(r_triangles.shape[0] + 1, 8, 8, 3) * 0.01 , requires_grad= True)

        print("Mesh path: ", self.mesh_path)
        print("Vertices shape: ", self.vertices.shape)
        print("Triangles shape: ", self.faces.shape)
        print("Opacity shape: ", self.opacity.shape)
        print("Texture shape: ", self.texture.shape)



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d
