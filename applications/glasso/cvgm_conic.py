import numpy as np
import random
from diffcp import solve_and_derivative
import ipdb
from applications.glasso.glasso_conic import write_glasso_cone_program


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
            theta_size = int(p*(p+1)/2)

            # write program in correct form and run it
            A, b, c, cone_dict = write_glasso_cone_program(cov_train, curr_alpha)
            x, y, s, derivative, adjoint_derivative = solve_and_derivative(A, b, c, cone_dict)
            theta = _vec2symmetric(x[:theta_size], p)

            print("=====")
            print(f"Training loss: {np.sum(cov_train * theta) - np.log(np.linalg.det(theta))}")

            cov_test = np.cov(test_data, rowvar=False)
            print(f"Test loss: {np.sum(cov_test * theta) - np.log(np.linalg.det(theta))}")
            dl_dtheta = cov_test - np.linalg.inv(theta)

            dc = np.zeros(c.shape)
            dc[-theta_size:] = 1
            dx, dy, ds = derivative(np.zeros(A.shape), np.zeros(b.shape), dc)
            dtheta_d_alpha = _vec2symmetric(dx[:theta_size], p)

            dl_dalpha = np.sum(dl_dtheta * dtheta_d_alpha)
            gradients.append(dl_dalpha)
        print(f"gradients: {gradients}")

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


