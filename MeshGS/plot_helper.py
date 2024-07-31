import matplotlib.pyplot as plt
import os
from PIL import Image
import re


GT = './GT_hotdog_zero_grad_rgb_bar_plots/'
results = './rgb_hotdog_zero_grad_rgb_bar_plots/'


def extract_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())

image_names = os.listdir(GT)
image_names = sorted(image_names, key = extract_number)
n_images = len(image_names)

images_per_figure = 2
n_figures = n_images // images_per_figure + 1


for fig_idx in range(n_figures):
    start_idx = fig_idx * images_per_figure
    end_idx = min(start_idx + images_per_figure, n_images)
    fig, axs = plt.subplots(nrows=images_per_figure, ncols=2, figsize=(10, images_per_figure * 5))
    
    for idx, image_name in enumerate(image_names[start_idx:end_idx]):
        img_GT = Image.open(os.path.join(GT, image_name))
        img_result = Image.open(os.path.join(results, image_name))
        print
        axs[idx, 0].imshow(img_GT)
        axs[idx, 0].set_title('GT')

        axs[idx, 1].imshow(img_result)
        epoch = re.search(r'\d+', image_name).group()
        axs[idx, 1].set_title('My result, epoch: {}'.format(epoch))


        axs[idx, 0].axis('off')
        axs[idx, 1].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"compare_rgb/compare_rgb__zero_grad_{fig_idx + 1}.png")
