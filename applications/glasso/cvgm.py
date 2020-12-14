import numpy as np
import random
from diffcp import solve_and_derivative


def gradient_descent_update(p, p_grad, eta=.1):
    return p - eta * p_grad


def cvgm(samples, prob, alpha0, K=10, p=0.7, tol=1e-4, gradient_update=gradient_descent_update):
    nsamples = samples.shape[0]
    sample_ixs = np.array(list(range(nsamples)))
    num_training = int(p*nsamples)
    training_ix_list = [random.sample(sample_ixs, num_training) for _ in range(K)]
    training_masks = [sample_ixs == training_ixs for training_ixs in training_ix_list]
    training_datasets = [samples[mask] for mask in training_masks]
    test_datasets = [samples[~mask] for mask in training_masks]

    diff = float('inf')
    curr_alpha = alpha0
    A, b, c = prob["A"], prob["b"], prob["c"]
    while diff > tol:
        solutions = []
        for training_data in training_datasets:
            x, y, s, derivative, adjoint_derivative = solve_and_derivative(A, b, c, training_data, curr_alpha)
            solutions.append(None)

        # calculate gradient of validation loss w.r.t. alpha
        dA, db, dc = adjoint_derivative(np.ones(x.shape), np.zeros(y.shape), np.zeros(s.shape))
        alpha_grad = None

        # update alpha
        new_alpha = gradient_update(curr_alpha, alpha_grad)
        diff = np.linalg.norm(new_alpha - curr_alpha)

    s = prob(samples, new_alpha)
    return new_alpha, s


