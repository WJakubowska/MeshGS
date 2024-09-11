from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_normal_consistency
from pytorch3d.loss import chamfer_distance
from src.coords_utils import find_uv_coordinates
from src.load_blender import load_blender_data
from src.visualization import plot_loss_psnr
from src.ray_utils import *
from src.metrics import *
from src.model import MeshGS, MLP
from tqdm import tqdm, trange
import torch.nn.functional as F
import numpy as np
import imageio
import trimesh
import torch 
import time
import os


device = torch.device("cpu")
np.random.seed(0)
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
DEBUG = False

def batchify_rays(mlp, vertices, offsets,  background,  faces,  opacity, texture, i_iter,  rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(mlp, vertices, offsets,  background, faces,  opacity, texture, i_iter,  rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(mlp, vertices, offsets, background, faces, opacity, texture, i_iter,  H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
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
    all_ret = batchify_rays(mlp, vertices, offsets,  background, faces, opacity, texture, i_iter, rays, chunk, **kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]



def render_path(mlp, vertices, offsets, background, faces, opacity, texture, i_iter,  render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, metricbase = None, mesh_target_path = None, epoch = None):

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
        rgb, disp, acc, _ = render(mlp, vertices, offsets, background, faces,  opacity, texture, i_iter,  H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        accs.append(acc.numpy())
        if i==0:
            print(f"rgb_shape: {rgb.shape}, disp_shape: {disp.shape}, acc_shape: {acc.shape}")

            if gt_imgs is not None and render_factor == 0 and metricbase is not None:
                psnr_metric = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
                target = torch.tensor(gt_imgs[i])
                ssim_metric =  calculate_ssim(rgb, target)
                lpips_metric = calculate_lpips(rgb, target)
                rgb_metrics = "PSNR: {:.3f}, SSIM: {:.3f}, LPIPS: {:.3f}".format(psnr_metric, ssim_metric, lpips_metric)
                print(rgb_metrics)
                mesh_target_trim = trimesh.load(mesh_target_path, force='mesh')
                # mesh = trimesh.Trimesh(vertices=vertices+offsets, faces=faces)
                verts_target = torch.tensor(mesh_target_trim.vertices, dtype=torch.float32)
                faces_target = torch.tensor(mesh_target_trim.faces, dtype=torch.int64)
                mesh_target = Meshes(verts=[verts_target], faces=[faces_target])

                mesh = Meshes(verts=[vertices.float() + offsets.float()], faces=[faces])
                cd_loss = calculate_chamfer_distance_loss(mesh, mesh_target)
                try:
                    mesh_target = mesh_target_trim
                    mesh = trimesh.Trimesh(vertices=vertices+offsets, faces=faces)
                    cd_medium = calculate_chamfer_distance_medium(mesh, mesh_target)
                except:
                    cd_medium = 852.852456
                mesh = Meshes(verts=[vertices.float() + offsets.float()], faces=[faces])
                try:
                    nc = mesh_normal_consistency(mesh)
                except:
                    nc = 852.852456
                geometry_metrics = " CD l: {:.3f}, CD m: {:.3f}, NC: {:.3f}".format(cd_loss, cd_medium , nc)
                text = str(epoch) + ": " + rgb_metrics + geometry_metrics
                save_metric(text, metricbase)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_MeshGS(args):
    
    input_dim = 8 * 8 * 3 * 3 * 3
    hidden_dim1 = 256
    hidden_dim2 = 128
    output_dim = 3


    model = MeshGS(mesh_path=args.mesh_path, 
                   n_subdivisions=args.n_subdivisions, 
                   train_vertices=args.train_vertices, 
                   texture_size=args.texture_size).to(device)
    # mlp = SimpleMLP(input_dim, hidden_dim1, hidden_dim2, output_dim)
    mlp = 0

    grad_vars = [model.faces, model.vertices, model.texture, model.opacity]
    offsets_params = [model.offsets, model.background] 


    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    optimizer_offsets = torch.optim.Adam(offsets_params, lr=args.lrate_offsets)


    # grad_vars = list(model.parameters())
    # Create optimizer

        # input_dim = 8 * 8 * 3 * 24 * 2

    # pierwszy to był 128 i 64 

    
    # optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.001)
    optimizer_mlp = 0
    # offsets_params = [model.offsets, model.background] + list(mlp.parameters())

    # the best is lr=1e-3
    # optimizer_offsets = 0
    start = 0
    
    ##########################

    render_kwargs_train = {
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'opacity_noise_std' : args.opacity_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        render_kwargs_train['ndc'] = False

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['opacity_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, optimizer_offsets,  model, mlp, optimizer_mlp



def raw2outputs(i_iter, N_rays, opacity_tabs, rgb_tabs, opacity_noise_std=0, white_bkgd=False):
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
        noise = 0.
        if opacity_noise_std > 0.:
            noise = noise = torch.randn(opacity_tabs.shape) * opacity_noise_std
        alpha = raw2alpha(opacity_tabs + noise) # [N_rays, N_samples]
    
    else:
        rgb = torch.ones(N_rays, 1, 3)
        alpha = torch.ones(N_rays, 1)

    
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights, -1) 
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    assert torch.all(torch.isclose(acc_map, torch.tensor(1.0), atol=1e-10) | (acc_map < 1.0)), "Values in acc_map exceed 1.0"
    assert torch.all(acc_map >= 0.0), "Values in acc_map are less than 0.0"

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
    

    assert torch.all(torch.isclose(rgb_map, torch.tensor(1.0), atol=1e-10) | (rgb_map < 1.0)), "Values in rgb_map exceed 1.0"
    assert torch.all(rgb_map >= 0.0), "Values in rgb_map are less than 0.0"

    
    return rgb_map, disp_map, acc_map, weights, depth_map

# def sh_encoding(rays_d, num_freqs):
#     frequencies = 2.0 ** torch.arange(num_freqs)
#     rays_d_expanded = rays_d.unsqueeze(-2)  
#     frequencies_expanded = frequencies.unsqueeze(-1).unsqueeze(-1) 
    
 
#     encoded = torch.cat([
#         torch.sin(frequencies_expanded * rays_d_expanded),
#         torch.cos(frequencies_expanded * rays_d_expanded)
#     ], dim=-1)
    

#     encoding = encoded.view(rays_d.shape[0], -1)
    
#     return encoding


def render_rays(mlp, vertices, offsets, background, faces, opacity, texture, i_iter, ray_batch, white_bkgd=False, opacity_noise_std=0.):
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
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
    """

    opacity_tabs, uv_coords = [], []
    N_rays = ray_batch.shape[0]
    # print("N_RAYS: ", N_rays)
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    background.zero_()
    # Ray tracking
    vertices_mesh = vertices + offsets


    mesh = trimesh.Trimesh(vertices=vertices_mesh.detach().numpy(), faces=faces)
    points, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=rays_o, ray_directions=rays_d)
    assert isinstance(mesh.ray, trimesh.ray.ray_pyembree.RayMeshIntersector), "The trimesh doesn't use pyembree."

    # Sort faces along the ray by distance
    intersections_by_ray = sort_faces_along_the_ray_by_distance(points, index_ray, index_tri, rays_o)

    # Create a dictionary {index_ray} : [faces_indices], {index_ray}: [points]
    # which maps each ray to the indices of the faces it intersects. If there are no intersections, assign None.
    intersects_faces_dict, intersects_points_dict = map_ray_to_intersections(intersections_by_ray, N_rays)

    # Assigning color and transparency to the intersected face
    unique_values, counts = np.unique(index_ray, return_counts=True)
    if len(points) == 0:
         rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(i_iter, N_rays, opacity_tabs, [], opacity_noise_std, white_bkgd)
        
    else: # Rays don't intersect the mesh anywhere
        max_valid_points = max(counts) 

        hit_tris = torch.full((N_rays, max_valid_points), torch.tensor(0))
        opacity_tabs = torch.full((N_rays,  max_valid_points), torch.tensor(float('-inf')))
        uv_coords = torch.zeros((N_rays, max_valid_points, 2))

        for index_ray, faces_indices in intersects_faces_dict.items():
            if faces_indices is None or len(faces_indices) == 0:
                continue
            
            points = intersects_points_dict[index_ray]
            sum_valid_points_per_ray = len(faces_indices)

            num_points = min(sum_valid_points_per_ray, max_valid_points)
            selected_faces = faces[faces_indices[:num_points]]
            vertex_indices = torch.unique(selected_faces).long()
            background[vertex_indices] = 1

            hit_tris[index_ray, :num_points] = torch.tensor(faces_indices[:num_points])
            uv = find_uv_coordinates(points[:num_points], faces_indices[:num_points], faces, vertices_mesh)
            uv_coords[index_ray, :num_points] = uv
            opacity_tabs[index_ray, :num_points] = opacity[faces_indices]            
        texture_tabs = texture[hit_tris.view(-1)]
        # print("Wymiary: ", texture_tabs.shape, rays_d.shape, uv_coords.shape)
        # print("uv: ", uv_coords.shape)
        uv_coords = uv_coords.view((-1, 1, 1, 2))
        # texture_size = texture_tabs.shape[1]
        # encoded_rays_d = sh_encoding(rays_d, num_freqs=4)
        # encoded_rays_d = rays_d
        # encoded_rays_d = (encoded_rays_d - encoded_rays_d.min()) / (encoded_rays_d.max() - encoded_rays_d.min())
        # print("SH: ", encoded_rays_d.shape)
        # 24 = 6 * num_freqs
        # encoded_rays_d  = encoded_rays_d .view(N_rays, 1, 1, 1, 1, 3)
        # uv_coords = uv_coords.view(N_rays, max_valid_points, 1, 1, 1, 1, 3)
        # print("uv: ", uv_coords.shape)
        # print("SH: ", encoded_rays_d.shape)
        # texture_tabs  = texture_tabs.view(N_rays, max_valid_points, texture_size, texture_size, 3).unsqueeze(-1).expand(-1, -1, -1, -1, -1, 3) 
        # print("Tutaj: ", texture_tabs.shape)
        # data = texture_tabs + encoded_rays_d
        # print("Data: ", data.shape)
        # data = data.unsqueeze(-1).expand(-1, -1, -1, -1,-1, -1, 3) + uv_coords
        # print("Data: ", data.shape)
        # print(data[:1])

        # rgb_raw = mlp(data)
        # print("RGB RAW: ", rgb_raw.shape)
        texture_tabs = texture_tabs.permute((0, 3, 1, 2))
        rgb_raw = torch.nn.functional.grid_sample(texture_tabs , uv_coords, align_corners=False).squeeze().view((N_rays, max_valid_points, 3)) 

        # 8 cech do MLP z 1 ukrytą + rays_d (można znormalizować) po sh encoding - jak mlp to bez grid sample albo kilka cech z grid_sample 
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(i_iter, N_rays, opacity_tabs, rgb_raw, opacity_noise_std, white_bkgd)
        

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret



def train(parser):

    loss_tab = []
    psnr_tab = []
    iter_tab = []

    
    args = parser.parse_args()
    print(args.expname)

    # Load Blender data
    images, poses, render_poses, hwf, i_split, acc_gt = load_blender_data(args.datadir, args.half_res, args.testskip)
    print('Loaded blender images: ', images.shape, poses.shape, render_poses.shape, args.datadir)
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
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, optimizer_offsets, model, mlp, optimizer_mlp = create_MeshGS(args)



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

    N_iters = 100000 + 1
    print('Begin')

    prev_param_values = {}

    start = start + 1
    points = trimesh.load("point_cloud.ply")
    points = points.vertices
    pcd_target = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
    print(pcd_target.shape)

    for i in trange(start, N_iters):
        time0 = time.time()
        model.train()
        # mlp.train()


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
        optimizer.zero_grad()
        optimizer_offsets.zero_grad()
        # optimizer_mlp.zero_grad()


        vertices = model.get_vertices()
        faces = model.get_faces()
        opacity = model.get_opacity()
        texture = model.get_texture()
        offsets = model.get_offsets()
        background = model.get_background()
        rgb, disp, acc, extras = render(mlp, vertices, offsets, background, faces, opacity, texture, i, H, W, K, chunk=args.chunk, rays=batch_rays,
                                                **render_kwargs_train)

        device1 = torch.device("cuda")
        vertices = vertices.float().to(device1)
        faces = faces.to(device1)
        offsets = offsets.float().to(device1)
        pcd_target = pcd_target.to(device1)
        _mesh = Meshes(verts=[vertices.float() + offsets.float()], faces=[faces])


        points = sample_points_from_meshes(_mesh, num_samples=pcd_target.shape[1])
        loss_chamfer, _ = chamfer_distance(points, pcd_target)
        


        img_loss = img2mse(rgb, target_s)
        mask_loss = img2BCE(torch.clamp(acc, min=0., max=1.), acc_target_s.float())
        l1_loss_rgb = l1_loss(rgb, target_s) 
        l2_loss = torch.mean((rgb - target_s) ** 2)
        loss = l1_loss_rgb + mask_loss + l2_loss 



        # loss = l1_loss_rgb + mask_loss + l2_loss + loss_lap + loss_normal
        psnr = mse2psnr(img_loss)


        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        
        loss.backward()
        optimizer.step()
        optimizer_offsets.step()
        # optimizer_mlp.step()


        for name, param in model.named_parameters():
            # print(name, param)
            if any(keyword in name for keyword in ['offsets', 'texture']):
                prev_value = prev_param_values.get(name)
                if prev_value is not None and torch.equal(prev_value, param):
                    print(f'not change value: {name}')
                prev_param_values[name] = param.clone().detach()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 10000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
            

        # decay_steps = args.lrate_decay * 10000
        # new_lrate = args.lrate_offsets * (decay_rate ** (global_step / decay_steps))
        # for param_group in optimizer_offsets.param_groups:
        #     param_group['lr'] = new_lrate
            # print("Zmiana, epoka: ", i, " lr: ", new_lrate)
        # ################################

        dt = time.time()-time0


        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            metricbase = os.path.join(basedir, expname, '{}_metrics.txt'.format(expname))

            with torch.no_grad():
                vertices = model.get_vertices()
                faces = model.get_faces()
                opacity = model.get_opacity()
                texture = model.get_texture()
                offsets = model.get_offsets()
                background = model.get_background()
                rgbs, disps = render_path(mlp, vertices, offsets, background, faces, opacity, texture, i,  render_poses, hwf, K, args.chunk, render_kwargs_test)
                rgbs_metric, disps_metric = render_path(mlp, vertices, offsets, background, faces, opacity, texture, i,  poses[:1], hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[:1], metricbase=metricbase, mesh_target_path=args.mesh_target_path, epoch=i)

            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))            
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)

    
        if i%args.i_print==0:
            plot_loss_psnr(basedir, expname, loss_tab, psnr_tab, iter_tab, loss.item(), psnr.item(), i)
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


