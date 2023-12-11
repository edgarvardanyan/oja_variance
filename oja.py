import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from p_tqdm import p_map
import json
import os


matplotlib.use('TkAgg')  # Use TkAgg backend for GUI display


class OjasRule:
    def __init__(self, initial_weights, ndim):
        if len(initial_weights) != ndim:
            raise ValueError(f"Initial weights should be of length {ndim}.")
        self.weights = np.array(initial_weights)
        self.ndim = ndim

    def fit(self, data, alpha):
        # Check shape
        if data.shape[1] != self.ndim:
            raise ValueError(f"Input data should have shape [N, {self.ndim}].")

        N = data.shape[0]

        for i in range(N):
            x = data[i, :]
            
            # Calculate output y = w^T x
            y = np.dot(self.weights, x)

            # Update weights using Oja's rule
            delta_w = alpha * y * (x - y * self.weights)
            self.weights += delta_w


def run_experiment(args):
    alpha, initial_weights, ndim, mean, cov, batches = args
    oja = OjasRule(initial_weights, ndim)
    for _ in range(batches):
        data = np.random.multivariate_normal(mean=mean, cov=cov, size=10000)
        oja.fit(data, alpha)
    return oja.weights


def dump_to_json(filename, alphas, variances):
    """
    Dump the alphas and their corresponding variances to a JSON file.
    
    :param filename: Name of the file to write data.
    :param alphas: List of alpha values.
    :param variances: List of variance values corresponding to each alpha.
    """
    data = {
        "alphas": alphas,
        "variances": [variance.tolist() for variance in variances]
    }
    
    with open(filename, 'w') as file:
        json.dump(data, file)


def load_from_json(filename):
    """
    Load alphas and their corresponding variances from a JSON file.
    
    :param filename: Name of the file to read data from.
    :return: alphas as a list and variances as a numpy array.
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    
    alphas = data["alphas"]
    variances = np.array(data["variances"])
    
    return alphas, variances


if __name__ == "__main__":
    ndim = 2
    samples = 100000
    initial_weights = [0, 1.0]
    corrs = [-0.3, -0.7, 0.3, 0.7]
    for corr in corrs:
        cov = np.array([[1, corr], [corr, 1]])
        mean = np.zeros(ndim)
        alphas = [0.00001, 0.00002, 0.00004, 0.00007, 0.0001, 0.0002, 0.0004, 0.0007, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.09, 0.128]
        batches = [200, 100, 50, 30, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]  # number of 10.000 size batches to be used at each experiment
        num_experiments = 500

        variances = []
        for alpha, batch in zip(alphas, batches):
            args = [(alpha, initial_weights, ndim, mean, cov, batch) for _ in range(num_experiments)]
            weight_history = np.array(p_map(run_experiment, args))
            variances.append(weight_history.var(axis=0))
        variances = np.array(variances)

        json_path = os.path.join(os.path.dirname(__file__), f"variances_({str(corr).replace('.', '_')}).json")

        dump_to_json(json_path, alphas, variances)
