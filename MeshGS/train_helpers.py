import torch 
import matplotlib.pyplot as plt
import os
from scipy.spatial import cKDTree
import trimesh
import numpy as np

def plot_rgb_bar_chart(rgb_map, i_iter, name='hotdog'):

    rgb_map = (rgb_map * 255).to(torch.uint8)
    red = rgb_map[:, 0].detach().numpy()
    green = rgb_map[:, 1].detach().numpy()
    blue = rgb_map[:, 2].detach().numpy()

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.hist(red, bins=256, color='red', alpha=0.6)
    plt.title('Red')
    plt.xlabel('Value')
    plt.ylabel('Number of pixels')
    plt.xlim(0, 255)

    plt.subplot(1, 3, 2)
    plt.hist(green, bins=256, color='green', alpha=0.6)
    plt.title('Green')
    plt.xlabel('Value')
    plt.ylabel('Number of pixels')
    plt.xlim(0, 255)

    plt.subplot(1, 3, 3)
    plt.hist(blue, bins=256, color='blue', alpha=0.6)
    plt.title('Blue')
    plt.xlabel('Value')
    plt.ylabel('Number of pixels')
    plt.xlim(0, 255)

    plt.tight_layout()

    output_dir = './{}_rgb_bar_plots/'.format(name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'rgb_bar_{}.png'.format(i_iter))
    plt.savefig(output_path)
    plt.close('all')

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

def save_metric(text, file):
    with open(file, 'a') as plik:
        plik.write(f"{text}\n\n")



def calculate_chamfer_distance(mesh1, mesh2):
    points1 = mesh1.sample(10000) 
    points2 = mesh2.sample(10000)

    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    dist1, _ = tree1.query(points2)
    dist2, _ = tree2.query(points1)

    cd = np.mean(dist1) + np.mean(dist2)
    return cd

def get_normals_at_sampled_points(mesh, sampled_points):
    tree = cKDTree(mesh.vertices)
    _, indices = tree.query(sampled_points)
    return mesh.vertex_normals[indices]

def normal_consistency(normals1, normals2):
    dot_product = np.einsum('ij,ij->i', normals1, normals2)
    return np.mean(dot_product)

def calculate_normal_consistency(mesh1, mesh2):
    points1 = mesh1.sample(10000) 
    points2 = mesh2.sample(10000)

    normals_sampled1 = get_normals_at_sampled_points(mesh1, points1)
    normals_sampled2 = get_normals_at_sampled_points(mesh2, points2)
    
    return normal_consistency(normals_sampled1, normals_sampled2)


