import numpy as np
import cvxpy as cp
import ipdb


def _vec2symmetric(a, dim):
    A = np.zeros((dim, dim))
    A[np.triu_indices_from(A)] = a
    A = A + A.T
    A[np.diag_indices_from(A)] /= 2
    return A


def gradient_descent_update(p, p_grad, eta=.01):
    return p - eta * p_grad


def cvgm_glasso(samples, alpha0, K=10, p=0.7, tol=1e-4, max_iters=20, gradient_update=gradient_descent_update):
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

        for training_data, test_data in zip(training_datasets, test_datasets):
            cov_train = np.cov(training_data, rowvar=False)
            p = cov_train.shape[1]

            X = cp.Variable((p, p), symmetric=True)
            alpha_cp = cp.Parameter(nonneg=True)
            alpha_cp.value = curr_alpha
            constraints = [X >> 0]
            objective = cp.Minimize(cp.sum(cp.multiply(cov_train, X)) - alpha_cp*cp.log_det(X))
            prob = cp.Problem(objective, constraints)
            prob.solve(requires_grad=True)
            theta = X.value
            # write program in correct form and run it

            print("=====")
            print(f"Alpha: {curr_alpha}")
            print(f"Training loss: {np.sum(cov_train * theta) - np.log(np.linalg.det(theta))}")

            cov_test = np.cov(test_data, rowvar=False)
            print(f"Test loss: {np.sum(cov_test * theta) - np.log(np.linalg.det(theta))}")
            dl_dtheta = cov_test - np.linalg.inv(theta)

            alpha_cp.delta = 1
            prob.derivative()
            dtheta_d_alpha = X.delta

            dl_dalpha = np.sum(dl_dtheta * dtheta_d_alpha)
            print(dl_dalpha)
            gradients.append(dl_dalpha)
        print(f"Gradient Average: {np.mean(gradients)}")
        print(f"Gradient Std: {np.std(gradients)}")

        # update alpha
        alpha_grad = np.mean(gradients)
        diff = np.linalg.norm(alpha_grad)
        curr_alpha = gradient_update(curr_alpha, alpha_grad)
        curr_alpha = max(curr_alpha, 0)

        print(alpha_grad, curr_alpha)

    # s = prob(samples, new_alpha)
    return curr_alpha


if __name__ == '__main__':
    import causaldag as cd
    from sklearn.covariance import GraphicalLassoCV

    d = cd.rand.directed_erdos(5, .5)
    g = cd.rand.rand_weights(d)
    samples = g.sample(100)

    cvgm_alpha = cvgm_glasso(samples, 1)
    print(cvgm_alpha)

    glcv = GraphicalLassoCV(alphas=[.1, .2, .3, .4, .5, .6, .7, .8, .9], cv=10)
    glcv.fit(samples)
    print(glcv.alpha_)


