# Author: Lee Fingerhut
import numpy as np

from misc import datagen, plot_confusion_mat, plot_X_y
from mlp import MyMLPClassifier
from mlxtend.classifier import Adaline
from pathlib import Path


def part_d_a_labels_fn(features):
    labels = np.where(features[:, 1] >= 0.01, 1, -1)
    return labels


if __name__ == "__main__":
    cwd = Path.cwd()
    output_dir = cwd.joinpath(f"figures/part_d_a")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_features, test_features, train_labels, test_labels = datagen(
        labels_fn=part_d_a_labels_fn, train_size=1000, test_size=1000, seed=13
    )
    plot_X_y(train_features, train_labels, output_dir.joinpath(f"train_data.png"))
    plot_X_y(test_features, test_labels, output_dir.joinpath(f"test_data.png"))

    clf: MyMLPClassifier = MyMLPClassifier(
        hidden_layer_sizes=(16, 8, 2), activation="relu", learning_rate_init=0.001, random_state=1, max_iter=300
    )
    clf.fit(train_features, train_labels)
    pred_test_labels = clf.predict(test_features)
    clf.visualize_input(test_features, output_dir)
    last_hidden_layer = clf._forward_pass_fast_visualize_input(test_features, clf.n_layers_ - 1)
    x_adaline = np.array([last_hidden_layer[0], last_hidden_layer[1]]).T

    test_labels[test_labels < 0] = 0
    classifier_adaline = Adaline(epochs=2, eta=0.01, random_seed=0, minibatches=len(test_labels) // 8)
    classifier_adaline.fit(x_adaline, test_labels)
    pred_test_labels = classifier_adaline.predict(x_adaline)

    test_labels[test_labels == 0] = -1
    pred_test_labels[pred_test_labels == 0] = -1
    accuracy = 100.0 * np.mean(test_labels == pred_test_labels)
    print(accuracy)

    plot_confusion_mat(test_labels, pred_test_labels, output_dir.joinpath("confusion.png"))
    plot_X_y(test_features, pred_test_labels, output_dir.joinpath(f"predictions.png"))

