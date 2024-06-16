from utils.display_sphere import *
from load_blender import load_blender_data
from run_meshGS_helpers import *
from tqdm import tqdm, trange
import torch 
import numpy as np
import os
import imageio
import time
import torch.nn.functional as F
import configargparse
import time
import trimesh
import matplotlib.pyplot as plt
from collections import defaultdict

def config_parser():

    
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--opacity_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize opacity output, 1e0 recommended')

    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='blender', 
                        help='options: blender')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_video",   type=int, default=10000, 
                        help='frequency of render_poses video saving')
    
    # mesh options
    parser.add_argument("--icosphere", action='store_true', 
                        help='if true, the sphere is an isosphere, if false, the sphere is a sphere')
    parser.add_argument("--mesh_from_file", action='store_true',
                        help='if true, mesh is load from file')
    parser.add_argument("--mesh_path", type=str,
                        help='specific mesh .ply or .obj file')
    
    parser.add_argument("--save_mesh_ply", action='store_true', 
                        help='if true, the mesh is save as ply after render')
  

    return parser



#################################################################################################################################
################################### Traing with NeRF ############################################################################
#################################################################################################################################

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
np.random.seed(0)
DEBUG = False

def batchify_rays(vertices, faces,  opacity, rgb_color, i_iter,  rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(vertices, faces,  opacity, rgb_color, i_iter,  rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(vertices, faces, opacity, rgb_color, i_iter,  H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      vertices: Mesh vertices.
      faces: Mesh faces.
      i_iter: Current number of iterations.
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(vertices, faces, opacity, rgb_color, i_iter, rays, chunk, **kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        # print("k ", k, " k_sh ", k_sh, " all_ret[k].shape ", all_ret[k].shape)
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(vertices, faces, opacity, rgb_color, i_iter,  render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    accs = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        rgb, disp, acc, _ = render(vertices, faces,  opacity, rgb_color, i_iter,  H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        accs.append(acc.numpy())
        if i==0:
            print(f"rgb_shape: {rgb.shape}, disp_shape: {disp.shape}, acc_shape: {acc.shape}")

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_MeshGS(args):

    model = MeshGS(mesh_icosphere = args.icosphere, mesh_from_file=args.mesh_from_file, mesh_path=args.mesh_path).to(device)

    grad_vars = list(model.parameters())

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    
    ##########################

    render_kwargs_train = {
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'opacity_noise_std' : args.opacity_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['opacity_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, model

def normalize(v):
        return v / torch.norm(v)

def calculate_uv(mesh_vertices, mesh_faces):
        vertices = mesh_vertices
        faces = mesh_faces
        uv_coords = []

        for face in faces: 
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]

            edge1 = torch.tensor([v1.x - v0.x, v1.y - v0.y, v1.z - v0.z])
            edge2 = torch.tensor([v2.x - v0.x, v2.y - v0.y, v2.z - v0.z])
            n = normalize(torch.cross(edge1, edge2))
            
            if n[0] != 0 and n[1] != 0:
                a = n[1]
                b = -n[0]
                c = 0
            else:
                a = n[2]
                b = 0
                c = -n[1]

            u = normalize(torch.tensor([b, -a, 0.]))
            v = torch.cross(n, u)
            uv_coords.append(torch.stack([u, v]))

        return torch.stack(uv_coords)

# def check_barycentric_coordinates(u, v, w):
#     if u >= 0 and v >= 0 and w >= 0:
#         if abs(u + v + w - 1.0) < 1e-6:
#             return True 
#         else:
#             raise ValueError("Sum of barycentric coordinates u + v + w must equal 1")
#     else:
#         raise ValueError("Barycentric coordinates u, v, w must be non-negative")


def calculate_barycentric_coordinates(point, vertices_A, vertices_B, vertices_C):
    # https://ceng2.ktu.edu.tr/~cakir/files/grafikler/Texture_Mapping.pdf
        v0 = vertices_B - vertices_A
        v1 = vertices_C - vertices_A
        v2 = point - vertices_A
        
        d00 = torch.dot(v0, v0)
        d01 = torch.dot(v0, v1)
        d11 = torch.dot(v1, v1)
        d20 = torch.dot(v2, v0)
        d21 = torch.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        # check_barycentric_coordinates(u, v, w)
        return u, v, w



def find_uv_coordinates(points, faces_indices, faces, vertices):
    uv_coords = []
    for point, face_idx in zip(points, faces_indices):
        face = faces[face_idx]
        face = face.long()
        A = vertices[face[0]]
        B = vertices[face[1]]
        C = vertices[face[2]]
        u, v, w = calculate_barycentric_coordinates(torch.tensor(point), A, B, C)
        uv_coords.append(torch.tensor([u, v]))
    return torch.stack(uv_coords)





def raw2outputs(N_rays, opacity_tabs, rgb_tabs, uv_coords, opacity_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, act_fn=F.sigmoid: act_fn(raw)

    if rgb_tabs != [] and opacity_tabs != []:
   
        rgb = torch.sigmoid(rgb_tabs)  # [N_rays, N_samples, 3]
        uv_coords = uv_coords.unsqueeze(0).float()
        image = rgb.unsqueeze(0).permute(0, 3, 1, 2).float()
        # print(uv_coords.shape, image.shape, image.permute(0, 3, 1, 2).shape)
        sampled_color = F.grid_sample(image, uv_coords, align_corners=True)  
        # print(sampled_color, sampled_color.permute(2, 3, 1, 0).squeeze().shape)
        rgb = sampled_color.permute(2, 3, 1, 0).squeeze()


        noise = 0.
        if opacity_noise_std > 0.:
            noise = noise = torch.randn(opacity_tabs.shape) * opacity_noise_std
    
        alpha = raw2alpha(opacity_tabs + noise) # [N_rays, N_samples]
    
    else:
        rgb = torch.ones(N_rays, 1, 3)
        alpha = torch.ones(N_rays, 1)

    
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    sum_weights = torch.sum(weights, dim=1)
    sum_weights_equal_to_one = torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-6)

    if sum_weights_equal_to_one == True:
        print(" Sum weights equal to one")

    depth_map = torch.sum(weights, -1) 
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map



def sort_faces_along_the_ray_by_distance(points, index_ray, index_tri, rays_o):
    intersections_by_ray = defaultdict(list)
    # calculate distance
    for point, ray_idx, tri_idx in zip(points, index_ray, index_tri):
        distance = torch.linalg.norm(torch.tensor(point) - rays_o[ray_idx])
        intersections_by_ray[ray_idx].append((distance, tri_idx, point))   
    #sort by distance
    for ray_idx in intersections_by_ray:
        intersections_by_ray[ray_idx].sort(key=lambda x: x[0])

    return intersections_by_ray



def map_ray_to_intersections(intersections_by_ray, N_rays):
    intersects_faces_dict = defaultdict(list) # {ray_idx}: [face_idx]
    intersects_points_dict = defaultdict(list)

    for ray_index in intersections_by_ray:
        sorted_face_indices = [face_idx for distance, face_idx, point in intersections_by_ray[ray_index]]
        sorted_point = [point for distance, face_idx, point in intersections_by_ray[ray_index]]
        intersects_faces_dict[ray_index] = sorted_face_indices
        intersects_points_dict[ray_index] = sorted_point

    intersects_faces_dict  = dict(intersects_faces_dict)
    additional_keys = {key: None for key in range(N_rays) if key not in intersects_faces_dict}
    intersects_faces_dict.update(additional_keys)
    intersects_faces_dict = dict(sorted(intersects_faces_dict.items()))

    intersects_points_dict = dict(intersects_points_dict)
    intersects_points_dict.update(additional_keys)
    intersects_points_dict = dict(sorted(intersects_points_dict.items()))


    return intersects_faces_dict, intersects_points_dict 


def render_rays(vertices, 
                faces,
                opacity, 
                rgb_color,
                i_iter,
                ray_batch,
                white_bkgd=False,
                opacity_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      vertices: Mesh vertices.
      faces: Mesh faces.
      i_iter: Current number of iterations.
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      white_bkgd: bool. If True, assume a white background.
      opacity_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
    """
    # pts [N_rays, N_samples, 3]

    opacity_tabs, rgb_tabs, uv_coords = [], [], []
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
   
    # Ray tracking
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    points, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=rays_o, ray_directions=rays_d)
    assert isinstance(mesh.ray, trimesh.ray.ray_pyembree.RayMeshIntersector), "The mesh doesn't use pyembree."

    # Sort faces along the ray by distance
    intersections_by_ray = sort_faces_along_the_ray_by_distance(points, index_ray, index_tri, rays_o)

    # Create a dictionary {index_ray} : [faces_indices], {index_ray}: [points]
    # which maps each ray to the indices of the faces it intersects. If there are no intersections, assign None.
    intersects_faces_dict, intersects_points_dict = map_ray_to_intersections(intersections_by_ray, N_rays)

    # Assigning color and transparency to the intersected face
    
    # Case 1: Rays don't intersect the mesh anywhere
    if len(points) == 0:
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(N_rays, opacity_tabs, rgb_tabs, opacity_noise_std, white_bkgd, pytest=pytest)
    
    # Case 2: Some rays intersected the mesh
    else:
        unique_values, counts = np.unique(index_ray, return_counts=True)
        max_valid_points = max(counts)

        for index_ray, faces_indices in intersects_faces_dict.items():
            if faces_indices == None:
                opacity_tabs.extend([torch.tensor(1)] * max_valid_points) 
                rgb_tabs.extend([torch.ones(3)]  * max_valid_points)
                uv_coords.extend([torch.zeros(2)] * max_valid_points)
            else:
                sum_valid_points_per_ray = len(faces_indices)
                diff = max_valid_points - sum_valid_points_per_ray
                opacity_tabs.extend(opacity[faces_indices])
                rgb_tabs.extend(rgb_color[faces_indices])
                points = intersects_points_dict[index_ray]
                uv = find_uv_coordinates(points, faces_indices, faces, vertices)
                assert rgb_color[faces_indices].shape[0] == uv.shape[0]
                uv_coords.extend(uv)
                if diff > 0:
                    opacity_tabs.extend([torch.tensor(1)] * diff)
                    rgb_tabs.extend([torch.ones(3)] * diff)
                    uv_coords.extend([torch.zeros(2)] * diff)  
        
        opacity_tabs = torch.stack(opacity_tabs).reshape(N_rays, max_valid_points) # [N_rays, N_samples]
        rgb_tabs = torch.stack(rgb_tabs).reshape(N_rays, -1, 3)       # [N_rays, N_samples, 3]
        uv_coords = torch.stack(uv_coords).reshape(N_rays, -1, 2)  #[N_rays, N_samples, 2]
        
        # rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(N_rays,opacity_tabs, rgb_tabs, opacity_noise_std, white_bkgd, pytest=pytest)
 
    
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(N_rays, opacity_tabs, rgb_tabs, uv_coords, opacity_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret



def save_mesh_as_file(vertices, faces, rgb_color, opacity, meshbase):
    rgb_color = torch.sigmoid(rgb_color).detach().numpy()
    opacity = torch.sigmoid(opacity).detach().numpy()
    rgb_color = (rgb_color * 255).astype(np.uint8)
    colors_with_opacity = np.concatenate((rgb_color, opacity.reshape(-1, 1)), axis=1)
    mesh = trimesh.Trimesh(vertices=vertices.detach().numpy(), faces=faces.detach().numpy(), face_colors=colors_with_opacity[:-1])
    mesh.export(meshbase + '.ply')



def plot_loss_psnr(basedir, expname, loss_tab, psnr_tab, iter_tab, loss, psnr, iter):
    loss_tab.append(loss)
    psnr_tab.append(psnr)
    iter_tab.append(iter)
    labels = ['Loss', 'PSNR']
    values = [loss_tab, psnr_tab]
    colors = ['orange', 'green']

    for value, label, color in zip(values, labels, colors):
        plt.figure()
        plt.plot(iter_tab, value, label=label, color=color, linewidth=2)
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(os.path.join(basedir, expname, 'plot_' + label.lower() + '.png'))
    plt.close('all')

def train():

    loss_tab = []
    psnr_tab = []
    iter_tab = []

    parser = config_parser()
    args = parser.parse_args()

    # Load Blender data
    images, poses, render_poses, hwf, i_split, acc_gt = load_blender_data(args.datadir, args.half_res, args.testskip)
    print('Loaded blender', images.shape, render_poses.shape, acc_gt.shape, hwf, args.datadir)
    i_train, i_val, i_test = i_split
    near = 2.
    far = 6.

    if args.white_bkgd:
        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    else:
        images = images[...,:3]

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, model = create_MeshGS(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to device
    render_poses = torch.Tensor(render_poses).to(device)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to device
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    N_iters = 300000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    prev_param_values = {}

    start = start + 1


    for i in trange(start, N_iters):
        time0 = time.time()
        model.train()


        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            acc_target = acc_gt[img_i]
            target = torch.Tensor(target).to(device)
            acc_target = torch.Tensor(acc_target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                acc_target_s = acc_target[select_coords[:, 0], select_coords[:, 1]]

        #####  Core optimization loop  #####
        # optimizer.zero_grad()

        vertices = model.get_vertices()
        faces = model.get_faces()
        opacity = model.get_opacity()
        rgb_color = model.get_rgb_color()

        
        rgb, disp, acc, extras = render(vertices, faces, opacity, rgb_color, i,  H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10,
                                                **render_kwargs_train)



        img_loss = img2mse(rgb, target_s)
        mask_loss = img2BCE(torch.clamp(acc, min=0., max=1.), acc_target_s.float())
        loss = img_loss + mask_loss
        # loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        
        loss.backward()
        optimizer.step()

        # for name, param in model.named_parameters():
        #     if any(keyword in name for keyword in ['faces', 'vertices', 'rgb_color', 'opacity']):
        #         prev_value = prev_param_values.get(name)
        #         if prev_value is not None and torch.equal(prev_value, param):
        #             print(f'not change value: {name}')
        #         else:
        #             print(f'change value: {name}')
        #         prev_param_values[name] = param.clone().detach()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0


        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                vertices = model.get_vertices()
                faces = model.get_faces()
                opacity = model.get_opacity()
                rgb_color = model.get_rgb_color()
                rgbs, disps = render_path(vertices, faces,  opacity, rgb_color, i,  render_poses, hwf, K, args.chunk, render_kwargs_test)


            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            meshbase = os.path.join(basedir, expname, '{}_mesh_{:06d}'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            # imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
            if args.save_mesh_ply == True:
                save_mesh_as_file(vertices, faces, rgb_color, opacity, meshbase)

    
        if i%args.i_print==0:
            plot_loss_psnr(basedir, expname, loss_tab, psnr_tab, iter_tab, loss.item(), psnr.item(), i)
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


if __name__=='__main__':
    torch.set_default_device("cpu")
    train()