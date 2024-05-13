from utils.display_sphere import *
from load_blender import load_blender_data
from run_meshGS_helpers import *
from tqdm import tqdm, trange
import openmesh as om
import torch 
import numpy as np
import os
import imageio
import time
import torch.nn.functional as F
import configargparse
import time
import trimesh
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
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

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
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')



    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_video",   type=int, default=10000, 
                        help='frequency of render_poses video saving')
    
    # ray tracking options
    parser.add_argument("--icosphere", action='store_true', 
                        help='if true, the sphere is an isosphere, if false, the sphere is a sphere')
    parser.add_argument("--coords", action='store_true', 
                        help='if true, barycentric coordinates are used for rendering')    
  


    return parser



#################################################################################################################################
################################### Traing with NeRF ############################################################################
#################################################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def batchify_rays(args, vertices, faces,  opacity, rgb_color, i_iter,  rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(args, vertices, faces,  opacity, rgb_color, i_iter,  rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(args, vertices, faces, opacity, rgb_color, i_iter,  H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
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
        # special case to render full image
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
    all_ret = batchify_rays(args, vertices, faces, opacity, rgb_color, i_iter, rays, chunk, **kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        # print("k ", k, " k_sh ", k_sh, " all_ret[k].shape ", all_ret[k].shape)
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(args, vertices, faces, opacity, rgb_color, i_iter,  render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        rgb, disp, acc, _ = render(args, vertices, faces,  opacity, rgb_color, i_iter,  H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_MeshGS(args):

    model = MeshGS(mesh_icosphere = args.icosphere, mesh_from_file=False, mesh_path='./data1/mesh_test.ply').to(device)

    grad_vars = list(model.parameters())
    model_fine = None

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    
    ##########################

    render_kwargs_train = {
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, model


def calculate_barycentric_coordinates(point, vertices_A, vertices_B, vertices_C):
    # https://ceng2.ktu.edu.tr/~cakir/files/grafikler/Texture_Mapping.pdf
        v0 = vertices_B - vertices_A
        v1 = vertices_C - vertices_A
        v2 = point - vertices_A
        
        d00 = torch.sum(v0 * v0, dim=1) # torch.dot() - iloczyn skalarny
        d01 = torch.sum(v0 * v1, dim=1) # (X*Y).sum(axis = 1) == torch.tensor([torch.dot(X[0], Y[0]),torch.dot(X[1], Y[1])])
        d11 = torch.sum(v1 * v1, dim=1)
        d20 = torch.sum(v2 * v0, dim=1)
        d21 = torch.sum(v2 * v1, dim=1)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return u, v, w


def check_if_point_is_in_triangle(point, vertices_A, vertices_B, vertices_C):
    u, v, w = calculate_barycentric_coordinates(point, vertices_A, vertices_B, vertices_C)
    mask = (v >= 0) & (w >= 0) & (u >= 0) & (v + w + u <= 1)
    if mask.any():
        idx = torch.nonzero(mask)[0]
        return u[idx] *  vertices_A[idx] + v[idx] * vertices_B[idx] + w[idx] * vertices_C[idx]
    
    return torch.empty(0)


def find_barycentric_coordinates(points, vertices, faces, opacity, rgb_color):
    N_rays, N_samples, _ = points.shape
    opacity_tabs = []
    rgb_tabs = []
    A = vertices[faces[:, 0]]
    B = vertices[faces[:, 1]]
    C = vertices[faces[:, 2]]
    points = points.view(-1, 3)
    for point in points:
            idx = check_if_point_is_in_triangle(point, A, B, C)
            if idx is not torch.empty(0):
                opacity_tabs.append(opacity[idx])
                rgb_tabs.append(rgb_color[idx])
                # print("opa:, " , opacity_tabs[0].shape)
                # print("rgb:, " , rgb_tabs[0].shape)
            else:
                assert False, "Triangle not found for point"
    opacity_tabs = torch.stack(opacity_tabs)
    opacity_tabs = opacity_tabs.view(N_rays, N_samples)
    rgb_tabs = torch.stack(rgb_tabs)
    rgb_tabs = rgb_tabs.view(N_rays, N_samples, 3)
    return opacity_tabs, rgb_tabs 



def raw2outputs(opacity_tabs, rgb_tabs, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
      # faces = faces.long()
    # # raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    raw2alpha = lambda raw, act_fn=F.sigmoid: act_fn(raw)
    # num_rays, num_samples = raw.shape[:2]


    rgb = torch.sigmoid(rgb_tabs) # jak sigmoid to zera w dodawaniu pkt  # [N_rays, N_samples, 3]
    # rgb = rgb_tabs

    # noise = 0.
    # if raw_noise_std > 0.:
    #     noise = torch.randn(raw[...,3].shape) * raw_noise_std


    # alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]

    # rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    # noise = 0.
    # if raw_noise_std > 0.:
        # noise = torch.randn(raw[...,3].shape) * raw_noise_std


    # alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    alpha = raw2alpha(opacity_tabs)
    
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights, -1) 
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map





def render_rays(args,
                vertices, 
                faces,
                opacity, 
                rgb_color,
                i_iter,
                ray_batch,
                white_bkgd=False,
                raw_noise_std=0.,
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
      raw_noise_std: ...
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
    # pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    opacity_tabs, rgb_tabs = [], []
    N_rays = ray_batch.shape[0]
    
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
   
   
    mesh = trimesh.Trimesh(vertices=vertices.cpu(), faces=faces.cpu())
    points, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=rays_o.cpu(), ray_directions=rays_d.cpu())
    assert isinstance(mesh.ray, trimesh.ray.ray_pyembree.RayMeshIntersector), "The mesh doesn't use pyembree."

    ray_to_tri_dict = defaultdict(list)
    for ray_index, tri_value in zip(index_ray, index_tri):
        ray_to_tri_dict[ray_index].append(tri_value)
    ray_to_tri_dict = dict(ray_to_tri_dict)
    additional_keys = {key: None for key in range(N_rays) if key not in ray_to_tri_dict}
    ray_to_tri_dict.update(additional_keys)

    if len(points) == 0:
        opacity_tabs = torch.tensor([opacity[-1] for _ in range(N_rays)]).unsqueeze(1) # [N_rays, 1]
        rgb_tabs =  rgb_color[-1].repeat(N_rays, 1, 1) # [N_rays, 1,  3] 

    else:
        unique_values, counts = np.unique(index_ray, return_counts=True)
        max_valid_points = max(counts)

        for value in ray_to_tri_dict.values():
            if value == None:
                opacity_tabs.extend([opacity[-1]] * max_valid_points) 
                rgb_tabs.extend([rgb_color[-1]] * max_valid_points)
            else:
                sum_valid_points_per_ray = len(value)
                diff = max_valid_points - sum_valid_points_per_ray
                opacity_tabs.extend(opacity[value])
                rgb_tabs.extend(rgb_color[value])
                if diff > 0:
                    opacity_tabs.extend([torch.tensor(0)] * diff)
                    rgb_tabs.extend([torch.zeros(3)] * diff)    

        opacity_tabs = torch.stack(opacity_tabs).reshape(N_rays, max_valid_points) # [N_rays, N_samples]
        rgb_tabs = torch.stack(rgb_tabs).reshape(N_rays, -1, 3)       # [N_rays, N_samples, 3]



    # if i_iter%500 == 0:
    # if i_iter == 1:
    #     try:
    #         plot_selected_points_on_sphere(points_tabs, vertices, faces, 'hot_dog_' + str(i_iter)+ '.html')
    #     except:
    #         print("Error during plot selected points on sphere")
    
    
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(opacity_tabs, rgb_tabs, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def save_mesh_as_file(vertices, faces, colors, opacity, filename):
    mesh = om.TriMesh()
    vertices = np.array(vertices.detach().cpu().numpy())
    colors = np.array(colors.detach().cpu().numpy())
    opacity = np.array(opacity.detach().cpu().numpy())
    for vertex in vertices:
        mesh.add_vertex(vertex)

    for face in faces:
        vertex_handles = [mesh.vertex_handle(idx) for idx in face]
        mesh.add_face(*vertex_handles)   

    mesh.request_face_colors()
    for face, color, op in zip(mesh.faces(), colors, opacity):
        mesh.set_color(face, np.append(color, op))

    om.write_mesh(filename + '.obj', mesh, face_color=True)
    om.write_mesh(filename + '.ply', mesh, face_color=True)




def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load Blender data
    images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
    print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
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

    # Move testing data to GPU
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

    # Move training data to GPU
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
            target = torch.Tensor(target).to(device)
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

        #####  Core optimization loop  #####
        optimizer.zero_grad()

        vertices = model.get_vertices()
        faces = model.get_faces()
        opacity = model.get_opacity()
        rgb_color = model.get_rgb_color()

        rgb, disp, acc, extras = render(args, vertices, faces, opacity, rgb_color, i,  H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10,
                                                **render_kwargs_train)




        img_loss = img2mse(rgb, target_s)
        # img_loss = torch.abs((rgb - target_s)).mean()
        
        loss = img_loss
        # print("loss ", img_loss)
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
                rgbs, disps = render_path(args, vertices, faces,  opacity, rgb_color, i,  render_poses, hwf, K, args.chunk, render_kwargs_test)

            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            meshbase = os.path.join(basedir, expname, '{}_mesh_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
            try:
                save_mesh_as_file(vertices, faces, rgb_color, opacity, meshbase)
            except:
                print("Save mesh failed")

    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


if __name__=='__main__':
 
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    train()