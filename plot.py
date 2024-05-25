import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import math
from helpers import _map_to_ordinal

COLORS = ['tab:orange', 'tab:red', 'tab:purple', 'tab:brown',
          'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

def plot_samples(results, name, num_trajectories=0, plot_dir="./plots"):
    if (results['dim_x'] == 2) and (results['dim_y'] ==1):
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        axes.scatter(results['data']['x_test'][:, 0], results['data']['x_test'][:, 1], results['data']['y_test'], label='Test points', c='tab:purple')
        axes.scatter(results['data']['x_train'][:, 0], results['data']['x_train'][:, 1], results['data']['y_train'], label='Training points', c='tab:green')
        axes.scatter(results['data']['x_test'][:, 0], results['data']['x_test'][:, 1], results['y_test_mean'], label='Gen', c='tab:orange')

        axes.set_xlabel('X-axis')
        axes.set_ylabel('Y-axis')
        axes.set_zlabel('Z-axis')
        if ('avg_nll' in results.keys()) and ('mae' in results.keys()):
            axes.set_title(name + ' MAE = {:.5f} NLL = {:.5f}'.format(results['mae'], results['avg_nll']))
        axes.grid()
        axes.legend()
    elif (results['dim_x'] == 1) and (results['dim_y'] == 1):
        x_ordering = None
        if 'x_ordering' in results['data']:
            x_ordering = results['data']['x_ordering']
        fig, axes = plt.subplots(1, 1, figsize=(7, 4))
        axes.scatter(_map_to_ordinal(results['data']['x_train'], x_ordering), results['data']['y_train'], label='Training points', c='black', marker='v')
        if 'x_true' in results['data']:
            axes.plot(results['data']['x_true'], results['data']['y_true'], label='True Function', c='black')

        num_ys = len(results['y_test'])

        # Plot mean and standard deviation.
        xs = _map_to_ordinal(np.array(results['data']['x_test'][: num_ys]), x_ordering)
        ys = np.array(results['y_test_mean'])
        ys_med = np.array(results['y_test_median'])
        stds = np.array(results['y_test_std'])
        lowers = np.array(results['y_test_lower'])
        uppers = np.array(results['y_test_upper'])
        idxs = np.argsort(xs)
        axes.plot(xs[idxs], ys_med[idxs], alpha=1.0, c=COLORS[1], label='Median', marker='.')
        axes.fill_between(xs[idxs], lowers[idxs], uppers[idxs], color=COLORS[0], alpha=0.4, label='Confidence')
        # Plot trajectories.
        if num_trajectories > 0:
            # axes.set_ylim([0, 100])  # hacked in for lynx plot
            xs = np.array(results['data']['x_test'])
            for i in range(num_trajectories):
                ys = np.array(results['y_test'])[:, i]
                axes.plot(xs, ys, label="sample {}".format(i), alpha=0.4)
        if ('avg_nll' in results.keys()) and ('mae' in results.keys()):
            axes.set_title(name + ' MAE = {:.5f} NLL = {:.5f}'.format(results['mae'], results['avg_nll']))
        axes.grid()
        axes.legend()
        if x_ordering is not None:
            plt.xticks(results['data']['x_true'][0::5], rotation=90)
    elif (results['dim_x'] == 1) and (results['dim_y'] > 1):
        x_ordering = None
        if 'x_ordering' in results['data']:
            x_ordering = results['data']['x_ordering']
        fig, axes = plt.subplots(results['dim_y'], 1, figsize=(7, results['dim_y'] * 4), sharex=True)
        for i in range(results['dim_y']):
            # plot training points
            axes[i].scatter(_map_to_ordinal(results['data']['x_train'], x_ordering), results['data']['y_train'][:, i], label='Training points', c='black', marker='v')
            # plot true function
            if 'x_true' in results['data']:
                axes[i].plot(results['data']['x_true'], results['data']['y_true'][:, i], label='True Function', c='black')
            xs = _map_to_ordinal(np.array(results['data']['x_test']), x_ordering)
            ys_med = np.array(results['y_test_median'])[:, i]
            lowers = np.array(results['y_test_lower'])[:, i]
            uppers = np.array(results['y_test_upper'])[:, i]
            idxs = np.argsort(xs)
            axes[i].plot(xs[idxs], ys_med[idxs], alpha=1.0, c=COLORS[1], label='Median', marker='.')
            axes[i].fill_between(xs[idxs], lowers[idxs], uppers[idxs], color=COLORS[0], alpha=0.4, label='Confidence')
            if ('avg_nll' in results.keys()) and ('mae' in results.keys()):
                axes[i].set_title(name + ' MAE = {:.3f} NLL = {:.3f}'.format(results['mae'][i], results['avg_nll']))
            elif 'mae' in results.keys():
                axes[i].set_title(name + ' MAE = {:.3f}'.format(results['mae'][i]))
            axes[i].grid()
            axes[i].legend()
        if x_ordering is not None:
            plt.xticks(results['data']['x_true'][0::5], rotation=90)
    else:
        print("Unsupported plot dimensions.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, name + '.pdf'), bbox_inches='tight')


def plot_images(results, name, plot_dir):
    size = int(math.sqrt(len(results['data']['y_true'])))
    true_image_np = np.uint8(results['data']['y_true'].reshape(size, size))
    generated_image_median_np = np.uint8(np.array(results['y_test_median']).reshape(size, size))
    os.makedirs(plot_dir, exist_ok=True)
    plt.imsave(os.path.join(plot_dir, name + '_true.png'), true_image_np, cmap='gray')
    plt.imsave(os.path.join(plot_dir, name + '_median.png'), generated_image_median_np, cmap='gray')

    training_image = Image.new('RGB', (size, size))
    background_color = (173, 216, 230)
    training_image.paste(background_color, box=(0, 0, size, size))
    for loc, value in zip(results['data']['x_train'], results['data']['y_train']):
        training_image.putpixel((loc[1], loc[0]), (value, value, value))
    training_image.save(os.path.join(plot_dir, name + '_training.png'))


def plot_heatmap(results, name, plot_dir, xs, ys):
    fig, axes = plt.subplots(1, 1, figsize=(7, 4))

    # cmap = plt.colormaps["plasma"]
    cmap = plt.colormaps["Reds"]
    # cmap = cmap.with_extremes(bad=cmap(0))
    heatmap = np.array(results['dist'])
    pcm = axes.pcolormesh(xs, ys, heatmap.T, norm="log", rasterized=True, cmap=cmap, shading='gouraud')
    axes.set_ylabel("y")
    axes.set_xlabel("x")
    fig.colorbar(pcm, label="p(y): Logits")

    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, name + '_heatmap.pdf'), bbox_inches='tight')