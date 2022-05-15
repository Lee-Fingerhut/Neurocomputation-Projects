# Author: Lee Fingerhut
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def datagen(labels_fn, train_size: int, test_size: int = 10000, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    size = train_size + test_size
    features = 2 * np.random.rand(size, 2) - 1.0
    labels = labels_fn(features)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, train_size=train_size / (train_size + test_size), random_state=seed
    )
    return train_features, test_features, train_labels, test_labels


def plot_confusion_mat(y: np.ndarray, y_pred: np.ndarray, save_as=None):
    cm = confusion_matrix(y, y_pred)
    plt.subplots()
    sns.heatmap(cm, fmt=".0f", annot=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.savefig(save_as)


def plot_X_y(x: np.ndarray, y_pred: np.ndarray, save_as=None):
    _, axs = plt.subplots(1, 1, figsize=(5, 5))
    s = axs.scatter(x[:, 0], x[:, 1], c=y_pred, cmap=plt.cm.Spectral, s=20)
    legend = axs.legend(*s.legend_elements(), title="data", loc="best")
    axs.add_artist(legend)
    axs.set_title("MLP Prediction")
    plt.savefig(save_as)


def layers_diagram(X, index_layer, layer, output_dir):
    for n, neuron in enumerate(layer):
        _, axs = plt.subplots(1, 1, figsize=(5, 5))
        s = axs.scatter(x=X[:, 0], y=X[:, 1], c=np.where(neuron == -1, -1, 1), cmap=plt.cm.Spectral, s=20,)
        legend = axs.legend(*s.legend_elements(), title="", loc="best",)
        axs.add_artist(legend)
        axs.set_title("Layer: " + str(index_layer) + " Neuron: " + str(n))
        plt.savefig(output_dir.joinpath(f"Layer:{index_layer}_Neuron:{n}.png"))
