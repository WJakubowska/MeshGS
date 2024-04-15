import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.triangles_utils import get_unique_triangles, get_triangles_as_indices
from icosphere import Icosphere
from triangle import Triangle
from utils.gs import setup_training_input
from ball import *


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model

class MeshGS(nn.Module):
    def __init__(self, N_rand, N_sample, mesh_icosphere = True,  use_viewdirs=False):
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
        self.N_rand = N_rand
        self.N_sample = N_sample
        self.rgb_color = None
        self.opacity = None
        self.vertices = None
        self.faces = None
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = self.inverse_sigmoid

        if mesh_icosphere == True:
            self.unique_vertices, self.traingles = self.create_mesh_icosphere()
        else: 
            self.unique_vertices, self.traingles = self.create_mesh_sphere()
        
        self.setup_training_input(self.unique_vertices, self.traingles)

    def create_mesh_icosphere(self, center_point = (0, 0, 0), radius = [2, 1], n_subdivisions = 2):
        self.mesh = Icosphere(n_subdivisions, center_point, radius)
        vertices, triangles = icosphere.vertices, icosphere.triangles
        unique_triangles = get_unique_triangles(triangles)
        unique_vertices = icosphere.get_all_vertices()
        triangles = get_triangles_as_indices(unique_vertices, triangles)
        # self.vertices = unique_vertices
        # self.faces = result_triangles
        return unique_vertices, traingles
    
    def create_mesh_sphere(self, n_slices = 10, n_stacks = 10):
        unique_vertices, triangles = uv_sphere(n_slices, n_stacks)
        triangles = get_triangles_as_indices(unique_vertices, triangles)
        # faces, features_dc, features_rest, opacity, vertices = setup_training_input(vertices, result_triangles)
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
        # We create random points inside the bounds of the synthetic Blender scenes
        # xyz = np.random.random((self.num_pts, 3)) * 2.6 - 1.3
        # points = xyz
        # shs = np.random.random((self.num_pts, 3)) / 255.0
        # colors=SH2RGB(shs)


        # fused_point_cloud = torch.tensor(np.asarray(points)).float()
        # fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float())
        # features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float()
        # features[:, :3, 0 ] = fused_color
        # features[:, 3:, 1:] = 0.0

        # opacities = self.inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float))

        coordinates_list = [(vertex.x, vertex.y, vertex.z) for vertex in unique_vertices]
        vertices_tensor = torch.tensor(coordinates_list, dtype=torch.float64, requires_grad=True)
    
        vertices_table = nn.Parameter(vertices_tensor)

        r_triangles = torch.tensor(result_triangles, dtype=torch.float64, requires_grad=False)
        # self.xyz = nn.Parameter(r_triangles.requires_grad_(False))  # odpowiednik to wierzchołki grafu
        # self.features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self.features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # self.opacity = nn.Parameter(opacities.requires_grad_(True))
        
        self.faces = torch.nn.Parameter(r_triangles, requires_grad=False)
        self.vertices = torch.nn.Parameter(vertices_tensor, requires_grad=True)
        # self.opacity = torch.nn.Parameter(torch.ones(self.N_rand, self.N_sample), requires_grad=True)
        # self.rgb_color = torch.nn.Parameter(torch.ones(self.N_rand, self.N_sample, 3), requires_grad=True)

        self.opacity = torch.nn.Parameter(torch.ones(r_triangles.shape[0]), requires_grad=True)
        self.rgb_color = torch.nn.Parameter(torch.ones(r_triangles.shape[0], 3), requires_grad=True)

        #print("Shape:")
        #print("vertices_table: ", vertices_table.shape)
        #print("xyz: ", self.xyz.shape)
        #print("features_dc: ", self.features_dc.shape)
        #print("features_rest: ", self.features_rest.shape)
        #print("opacity: ", self.opacity.shape)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.vertices = None
        self.faces = None

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)


    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs   

    def get_vertices(self):
        return self.vertices

    def get_faces(self):
        return self.faces    



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
