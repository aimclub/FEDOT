import seaborn as sns
from matplotlib import pyplot as plt


def visualise_gradcam(att_maps,
                      median_sample,
                      figsize,
                      cmap,
                      **kwargs):
    # matplotlib.use('TKagg')

    if figsize is None:
        figsize = (12, 4)
    if att_maps[0].ndim == 3:
        att_maps[0] = att_maps[0].mean(1)
    if att_maps[1].ndim == 3:
        att_maps[1] = att_maps[1].mean(1)

    idx_plot = list(range(2 + len(median_sample)))
    fig, axs = plt.subplots(len(idx_plot), 1,
                            figsize=figsize,
                            sharex=True,
                            **kwargs)
    for idx, class_number in enumerate(median_sample):
        axs[idx].set_title(f'Median sample of {class_number}')
        sns.lineplot(median_sample[class_number].reshape(-1, 1), ax=axs[idx])

    axs[idx_plot[-2]].set_title('Observed Variables')
    axs[idx_plot[-1]].set_title('Attention by the time')

    sns.heatmap(att_maps[0].numpy(), cbar=False, cmap=cmap, ax=axs[idx_plot[-2]])
    sns.heatmap(att_maps[1].numpy(), cbar=False, cmap=cmap, ax=axs[idx_plot[-1]])
    fig.tight_layout()
    plt.show()
