import numpy as np
from skimage.transform import resize
from matplotlib import pyplot as plt
from skimage.util import view_as_windows
from sklearn.feature_extraction.image import extract_patches_2d


def self_attention(w_k, w_q, image, indexes, k):
    """
    Computes the A matrix for the self-attention module
    and sums its columns as done in
    Y. Tang, D. Nguyen, e D. Ha, «Neuroevolution of Self-Interpretable Agents», arXiv:2003.08165 [cs], mar. 2020 http://arxiv.org/abs/2003.08165

    :w_k: The Key matrix
    :w_q: The Query matrix
    :image: The image to perform attention on
    :indexes: The indexes of the locations (x, y)
    :k: The number of patches to consider
    :returns: A list of coordinates (x, y) of the top-k important patches
    """
    data = np.concatenate([image, np.ones((len(image), 1))], axis=-1)
    a = 1/np.sqrt(data.shape[1]) * np.dot(np.dot(data, w_k), np.dot(data, w_q).T)
    a = np.exp(a) / np.sum(np.exp(a), axis=1)
    assert len(a.shape) == 2

    voting = np.sum(a, axis=0)
    sorted_idx = np.argsort(voting)[:-(k+1):-1]
    """
    for i in sorted_idx:
        plt.figure()
        plt.imshow(image[i].reshape(5, 5, 3))
        plt.show()
    """
    return indexes[sorted_idx]


def make_patches(image, config):
    """
    Create patches from an image

    :image: The image to decompose into patches
    :config: A dictionary with the configuration
    :returns: A list of patches
    """
    patch_size = config["attention"]["patch_width"]
    """
    all_patches = view_as_windows(
        image,
        patch_size,
        config["attention"]["stride"]
    )
    """
    patches = extract_patches_2d(image, [patch_size, patch_size])
    s = int(np.sqrt(patches.shape[0]))
    stride = config["attention"]["stride"]
    patches = patches.reshape(s, s, -1)
    idx = np.arange(0, s, 1)
    indexes = []
    for i in idx:
        row = []
        for j in idx:
            row.append([i, j])
        indexes.append(row)
    indexes = np.array(indexes)
    patches = patches[::stride, ::stride]
    indexes = indexes[::stride, ::stride, ::-1]  # fmt to x, y
    s = patches.shape[0] * patches.shape[1]
    patches = patches.reshape(s, -1)
    indexes = indexes.reshape(s, -1)

    return patches, indexes


def build_features(indices, config):
    """
    Builds features from the indices of the patches

    :indices: The indices of the patches
    :config: The dictionary containing the config
    :returns: A 1d list of (x, y) coordinates
    """
    return indices.flatten()


def convert_obs(obs, config):
    obs = resize(np.array(obs[config["env"]["vertical_offset"]:], float) / 255, (config["env"]["obs_height"], config["env"]["obs_width"]), anti_aliasing=True)
    return obs


