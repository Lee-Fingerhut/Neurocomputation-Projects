# Author: Lee Fingerhut
"""
In order to visualize the activation for a given input, we override sklearn's MLPClassifier
and simply add methods that saves the activations during thr forward pass, which are highly
based on the existing forward method of the classifier: 'MLPClassifier._forward_pass_fast'.
"""
import numpy as np

from misc import layers_diagram
from pathlib import Path
from sklearn.neural_network._multilayer_perceptron import (
    MLPClassifier,
    ACTIVATIONS,
    safe_sparse_dot,
)


class MyMLPClassifier(MLPClassifier):
    def visualize_input(self, X: np.ndarray, output_dir: Path):
        for layer in range(2, self.n_layers_):
            layer_i = self._forward_pass_fast_visualize_input(X, layer)
            layers_diagram(X, layer - 1, layer_i, output_dir)

    def _forward_pass_fast_visualize_input(self, X, layer=None):
        """Based on '_forward_pass_fast'."""
        X = self._validate_data(X, accept_sparse=["csr", "csc"], reset=False)

        # Initialize first layer
        activation = X

        # Forward propagate
        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(layer - 1):
            activation = safe_sparse_dot(activation, self.coefs_[i])
            activation += self.intercepts_[i]
            if i != layer - 2:
                hidden_activation(activation)

        if activation.shape[1] > 1:
            return [self._label_binarizer.inverse_transform(activation[:, i]) for i in range(activation.shape[1])]

        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activation)

        return self._label_binarizer.inverse_transform(activation)
