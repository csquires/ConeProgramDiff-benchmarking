import numpy as np
import cvxpy as cp
import ipdb
from tqdm import tqdm


def _vec2symmetric(a, dim):
    A = np.zeros((dim, dim))
    A[np.triu_indices_from(A)] = a
    A = A + A.T
    A[np.diag_indices_from(A)] /= 2
    return A


def gradient_descent_update(p, p_grad, eta=.001):
    return p - eta * p_grad


def cvgm_glasso(samples, alpha0, K=10, p=0.7, tol=1e-4, max_iters=100, gradient_update=gradient_descent_update):
    nsamples = samples.shape[0]
    num_training = int(p*nsamples)

    training_masks = [np.zeros(nsamples, dtype=bool) for _ in range(K)]
    for mask in training_masks:
        mask[:num_training] = True
        np.random.shuffle(mask)

    training_datasets = [samples[mask] for mask in training_masks]
    test_datasets = [samples[~mask] for mask in training_masks]

    diff = float('inf')
    curr_alpha = alpha0
    num_iter = 0
    while diff > tol and num_iter < max_iters:
        num_iter += 1
        gradients = []

        print(f"=== Iteration {num_iter}. alpha={curr_alpha} ====")
        for training_data, test_data in tqdm(zip(training_datasets, test_datasets), total=K):
            cov_train = np.cov(training_data, rowvar=False)
            p = cov_train.shape[1]

            X = cp.Variable((p, p), PSD=True)
            alpha_cp = cp.Parameter(nonneg=True)
            alpha_cp.value = curr_alpha
            constraints = [X >> 0]
            cov_train_cp = cp.Parameter((p, p), PSD=True)
            cov_train_cp.value = cov_train
            objective = cp.Minimize(cp.sum(cp.multiply(cov_train, X)) - cp.log_det(X) + alpha_cp*cp.pnorm(X, 1))
            prob = cp.Problem(objective, constraints)
            prob.solve(requires_grad=True)
            theta = X.value

            # print(f"Training loss: {.5*np.sum(cov_train * theta) - .5*np.log(np.linalg.det(theta)) + p/2 * np.log(2*np.pi)}")

            cov_test = np.cov(test_data, rowvar=False)
            # print(f"Test loss: {np.sum(cov_test * theta) - np.log(np.linalg.det(theta))}")
            # print(f"Test loss: {.5*np.sum(cov_test * theta) - .5*np.log(np.linalg.det(theta)) + p/2 * np.log(2*np.pi)}")
            dl_dtheta = cov_test - np.linalg.inv(theta)

            alpha_cp.delta = 1
            prob.derivative()
            dtheta_d_alpha = X.delta

            X.gradient = cov_test - np.linalg.inv(theta)
            prob.backward()
            dl_dalpha = alpha_cp.gradient
            # print("Method 1:", dl_dalpha)
            dl_dalpha = np.sum(dl_dtheta * dtheta_d_alpha)
            # print("Method 2:", dl_dalpha)
            gradients.append(dl_dalpha)

        print(f"Gradient Average: {np.mean(gradients)}")
        print(f"Gradient Std: {np.std(gradients)}")

        # update alpha
        alpha_grad = np.mean(gradients)
        diff = np.linalg.norm(alpha_grad)
        curr_alpha = gradient_update(curr_alpha, alpha_grad)
        curr_alpha = max(curr_alpha, 0)

    # solve again
    X = cp.Variable((p, p), symmetric=True)
    constraints = [X >> 0]
    objective = cp.Minimize(cp.sum(cp.multiply(cov_train, X)) - curr_alpha*cp.log_det(X))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    theta = X.value
    return curr_alpha, theta


if __name__ == '__main__':
    import causaldag as cd
    from sklearn.covariance import GraphicalLassoCV

    d = cd.rand.directed_erdos(6, .2)
    g = cd.rand.rand_weights(d)
    samples = g.sample(100)

    cvgm_alpha, cvgm_theta = cvgm_glasso(samples, alpha0=.1, K=20, max_iters=5)
    print(cvgm_alpha)

    glcv = GraphicalLassoCV(alphas=[.01, .02, .05, .1, .15], cv=10)
    glcv.fit(samples)

    print(f"Selected alpha from standard CV: {glcv.alpha_}")
    print(f"Alphas: {glcv.cv_alphas_}")
    print(f"Average log likelihoods: {glcv.grid_scores_.mean(axis=1)}")

    heldout_samples = g.sample(1000)
    cov_heldout = np.cov(heldout_samples, rowvar=False)
    p = cov_heldout.shape[0]
    glcv_theta = np.linalg.inv(glcv.covariance_)
    print(f"CVGM loss: {.5*np.sum(cov_heldout * cvgm_theta) - .5*np.log(np.linalg.det(cvgm_theta)) + p/2 * np.log(2*np.pi)}")
    print(f"CV loss: {.5*np.sum(cov_heldout * glcv_theta) - .5*np.log(np.linalg.det(glcv_theta)) + p/2 * np.log(2*np.pi)}")

    glcv_edges = {frozenset({i, j}) for i, j in zip(*np.nonzero(~np.isclose(glcv_theta, 0))) if i != j}
    cvgm_edges = {frozenset({i, j}) for i, j in zip(*np.nonzero(~np.isclose(cvgm_theta, 0))) if i != j}
    print("False negatives in GLCV", len(d.moral_graph().edges - glcv_edges))
    print("False negatives in CVGM", len(d.moral_graph().edges - cvgm_edges))
    print("False positives in GLCV", len(glcv_edges - d.moral_graph().edges))
    print("False positives in CVGM", len(cvgm_edges - d.moral_graph().edges))
