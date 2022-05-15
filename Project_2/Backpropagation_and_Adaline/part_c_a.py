# Author: Lee Fingerhut
import numpy as np

from misc import datagen, plot_confusion_mat, plot_X_y
from mlp import MyMLPClassifier
from pathlib import Path


def part_c_a_labels_fn(features):
    labels = np.where(features[:, 1] >= 0.01, 1, -1)
    return labels


if __name__ == "__main__":
    cwd = Path.cwd()
    output_dir = cwd.joinpath(f"figures/part_c_a")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_features, test_features, train_labels, test_labels = datagen(
        labels_fn=part_c_a_labels_fn, train_size=1000, test_size=1000, seed=5
    )

    clf = MyMLPClassifier(
        hidden_layer_sizes=(16, 8, 4), activation="relu", learning_rate_init=0.1, random_state=1, max_iter=300
    ).fit(train_features, train_labels)
    pred_test_labels = clf.predict(test_features)
    clf.visualize_input(test_features, output_dir)
    accuracy = 100.0 * np.mean(test_labels == pred_test_labels)
    print(accuracy)

    plot_confusion_mat(test_labels, pred_test_labels, output_dir.joinpath("confusion.png"))
    plot_X_y(test_features, pred_test_labels, output_dir.joinpath(f"predictions.png"))
