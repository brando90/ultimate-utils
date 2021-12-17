import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


def plot_images(images, labels, preds=None):
    """
    ref:
        - https://gist.github.com/MattKleinsmith/5226a94bad5dd12ed0b871aed98cb123
    """

    assert len(images) == len(labels) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, 0, :, :], interpolation='spline16', cmap='gray')

        label = str(labels[i])
        if preds is None:
            xlabel = label
        else:
            pred = str(preds[i])
            xlabel = "True: {0}\nPred: {1}".format(label, pred)

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def visualize_some_samples(dataset: Dataset, shuffle=False, num_workers=4, pin_memory=False):
    sample_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=9,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory)
    """
    """
    from uutils.plot.image_visualization import plot_images
    data_iter = iter(sample_loader)
    images, labels = data_iter.next()
    X = images.numpy()
    plot_images(X, labels)

# - test

def visualize_data_test():
    pass