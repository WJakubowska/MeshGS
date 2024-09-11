from collections import defaultdict
import numpy as np
import torch

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
