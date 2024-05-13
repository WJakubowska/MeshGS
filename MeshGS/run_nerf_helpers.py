import torch
import numpy as np
from utils.triangles_utils import  get_triangles_as_indices
from icosphere import Icosphere
from ball import *
import trimesh
from vertex import Vertex

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Model

class MeshGS(torch.nn.Module):
    def __init__(self, mesh_icosphere = True, mesh_from_file = False, mesh_path = None):
        super(MeshGS, self).__init__()
        # self.active_sh_degree = 0
        # self.max_sh_degree = 3
        # self._xyz = torch.empty(0)
        # self._features_dc = torch.empty(0)
        # self._features_rest = torch.empty(0)
        # self._opacity = torch.empty(0)
        # self.max_radii2D = torch.empty(0)
        # self.xyz_gradient_accum = torch.empty(0)
        # self.denom = torch.empty(0)
        # self.optimizer = None
        # self.percent_dense = 0
        # self.spatial_lr_scale = 0

        self.rgb_color = None
        self.opacity = None
        self.vertices = None
        self.faces = None
        self.background = None
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = self.inverse_sigmoid
        self.mesh_path = mesh_path

        if mesh_from_file == True:
            if mesh_path is not None: 
                mesh = trimesh.load(mesh_path, force='mesh')
                vertex_list = []
                for vertex in mesh.vertices:
                    vertex = Vertex(vertex[0], vertex[1], vertex[2])
                    vertex_list.append(vertex)
                vertices = vertex_list
                self.unique_vertices = vertices
                self.traingles = mesh.faces

                
            else:
                raise ValueError("Path to mesh file is not provided")
        else:
            if mesh_icosphere == True:
                self.unique_vertices, self.traingles = self.create_mesh_icosphere()
            else: 
                self.unique_vertices, self.traingles = self.create_mesh_sphere()
        

        self.setup_training_input(self.unique_vertices, self.traingles)

    def create_mesh_icosphere(self, center_point = (0, 0, 0), radius = [2, 1], n_subdivisions = 2):
        mesh = Icosphere(n_subdivisions, center_point, radius)
        vertices, triangles = mesh.vertices, mesh.triangles
        unique_vertices = mesh.get_all_vertices()
        triangles = get_triangles_as_indices(unique_vertices, triangles)
        return unique_vertices, triangles
    
    def create_mesh_sphere(self, n_slices = 10, n_stacks = 10):
        print("Kuleczka")
        unique_vertices, triangles = uv_sphere(n_slices, n_stacks)
        triangles = get_triangles_as_indices(unique_vertices, triangles)
        return unique_vertices, triangles

    def get_vertices(self):
        return self.vertices

    def get_faces(self):
        return self.faces 

    def get_opacity(self):
        return self.opacity
    
    def get_rgb_color(self):
        return self.rgb_color

    def inverse_sigmoid(self, x):
        return torch.log(x/(1-x))


    def setup_training_input(self, unique_vertices, result_triangles):


        coordinates_list = [(vertex.x, vertex.y, vertex.z) for vertex in unique_vertices]
        vertices_tensor = torch.tensor(coordinates_list, dtype=torch.float64, requires_grad=False) 


        r_triangles = torch.tensor(result_triangles, dtype=torch.float64, requires_grad=False)

        self.faces = torch.nn.Parameter(r_triangles, requires_grad=False)
        self.vertices = torch.nn.Parameter(vertices_tensor, requires_grad=False)
        # self.opacity = torch.nn.Parameter(torch.ones(self.N_rand, self.N_sample), requires_grad=True)
        # self.rgb_color = torch.nn.Parameter(torch.ones(self.N_rand, self.N_sample, 3), requires_grad=True)

        self.opacity = torch.nn.Parameter(torch.zeros(r_triangles.shape[0] + 1), requires_grad=True)
        self.rgb_color = torch.nn.Parameter(torch.zeros(r_triangles.shape[0] + 1, 3), requires_grad=True)

        print("File name: ", self.mesh_path)
        print("Vertices: ", self.vertices.shape)
        print("Triangles: ", self.faces.shape)
        print("Opacity: ", self.opacity.shape)
        print("RGB: ", self.rgb_color.shape)



    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    



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
