import torch 

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
            else:
                assert False, "Triangle not found for point"
    opacity_tabs = torch.stack(opacity_tabs)
    opacity_tabs = opacity_tabs.view(N_rays, N_samples)
    rgb_tabs = torch.stack(rgb_tabs)
    rgb_tabs = rgb_tabs.view(N_rays, N_samples, 3)
    return opacity_tabs, rgb_tabs 
