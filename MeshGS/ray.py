from ray_tracing.rays import Rays
from ray_tracing.mesh import Mesh
import torch
import numpy as np


def find_intersection_points_with_mesh(
    plot: bool = True,
):


    rays = Rays(
        origin=torch.load('rays_origins.pt'),
        direction=torch.load('rays_directions.pt'),
    )

    """Test finding intersection points with mesh."""

    # Two mesh available to test
    vertices = torch.load('vertices.pt').double()
    faces = torch.load('faces.pt')

    mesh_two_icospheres = Mesh(
        vertices=vertices,
        faces=faces
    )

    out = rays.find_intersection_points_with_mesh(
        mesh=mesh_two_icospheres,
        plot=plot
    )

    return out

# From NeRF

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


def random_rotation_matrix():
    angle = np.random.uniform(0, 2 * np.pi)
    axis = np.random.rand(3) - 0.5
    axis /= np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    return np.array([[a**2 + b**2 - c**2 - d**2, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a**2 + c**2 - b**2 - d**2, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a**2 + d**2 - b**2 - c**2]])


def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d

# def get_rays_np(H, W, focal, c2w):
#     c2w = None
#     i, j = np.meshgrid(np.arange(W, dtype=np.float32),
#                        np.arange(H, dtype=np.float32), indexing='xy')

#     dirs = np.stack([(i-W*0.5)/focal, -(j-H*0.5)/focal, -np.ones_like(i)], -1)

#     if c2w is None:
#         c2w_list = np.array([random_rotation_matrix() for _ in range(dirs.shape[0])])
#     else:
#         c2w_list = np.array([c2w for _ in range(dirs.shape[0])])

#     rays_o = np.einsum('ijk,ikl->ijl', dirs, c2w_list[:, :3, :3])
#     rays_o += c2w_list[:, None, :3, -1] 

#     rays_d = np.sum(dirs[..., np.newaxis, :] * c2w_list[:, :3, :3], -1)

#     return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = tf.stack([o0, o1, o2], -1)
    rays_d = tf.stack([d0, d1, d2], -1)

    return rays_o, rays_d