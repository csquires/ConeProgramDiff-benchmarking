import numpy as np
import cvxpy as cp
import ipdb
from tqdm import tqdm
from termcolor import colored


def _vec2symmetric(a, dim):
    A = np.zeros((dim, dim))
    A[np.triu_indices_from(A)] = a
    A = A + A.T
    A[np.diag_indices_from(A)] /= 2
    return A


def gradient_descent_update(p, fp, dfdp, eta=.01, step_num=None):
    step = -eta/np.sqrt(step_num+1) * dfdp
    print(f"Taking gradient step of {-eta * dfdp}")
    return p + step


def newton_update(p, fp, dfdp, alpha=0.4, step_num=None):
    step = -alpha * fp / dfdp
    if np.isnan(step):
        ipdb.set_trace()
    print(f"Taking newton step of {step}")
    return p + step


def newton_update_decay(p, fp, dfdp, alpha=0.8, step_num=None):
    return p - alpha**step_num * fp / dfdp


def generate_partitions(samples, K=10, p=0.7):
    nsamples = samples.shape[0]
    num_training = int(p*nsamples)

    training_masks = [np.zeros(nsamples, dtype=bool) for _ in range(K)]
    for mask in training_masks:
        mask[:num_training] = True
        np.random.shuffle(mask)

    training_datasets = [samples[mask] for mask in training_masks]
    test_datasets = [samples[~mask] for mask in training_masks]

    return training_datasets, test_datasets


def grid_search_cv(samples, alphas, K=10, p=0.7):
    training_datasets, test_datasets = generate_partitions(samples, K=K, p=p)

    test_losses = np.empty((len(alphas), K))
    for alpha_ix, alpha in enumerate(alphas):
        for data_ix, (training_data, test_data) in enumerate(tqdm(zip(training_datasets, test_datasets), total=K)):
            cov_train = np.cov(training_data, rowvar=False)
            p = cov_train.shape[1]
            X = cp.Variable((p, p), PSD=True)
            constraints = [X >> 0]
            cov_train_cp = cp.Parameter((p, p), PSD=True)
            cov_train_cp.value = cov_train
            objective = cp.Minimize(cp.sum(cp.multiply(cov_train, X)) - cp.log_det(X) + alpha*cp.pnorm(X, 1))
            prob = cp.Problem(objective, constraints)
            prob.solve()
            theta = X.value

            cov_test = np.cov(test_data, rowvar=False)
            test_loss = np.sum(cov_test * theta) - np.log(np.linalg.det(theta))
            test_losses[alpha_ix, data_ix] = test_loss

    avg_losses = test_losses.mean(axis=1)
    min_ix = np.argmin(avg_losses)
    best_alpha = alphas[min_ix]

    cov_train = np.cov(samples, rowvar=False)
    p = cov_train.shape[1]
    X = cp.Variable((p, p), PSD=True)
    constraints = [X >> 0]
    cov_train_cp = cp.Parameter((p, p), PSD=True)
    cov_train_cp.value = cov_train
    objective = cp.Minimize(cp.sum(cp.multiply(cov_train, X)) - cp.log_det(X) + best_alpha*cp.pnorm(X, 1))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    theta = X.value

    return alphas[min_ix], theta


def cvgm_glasso(samples, alpha0, min_alpha=.01, K=10, sample_fraction=0.7, tol=1e-4, max_iters=100, gradient_update=gradient_descent_update, cov_heldout=None):

    diff = float('inf')
    curr_alpha = alpha0
    num_iter = 0
    alpha_path = []
    test_loss_path = []
    while diff > tol and num_iter < max_iters:
        training_datasets, test_datasets = generate_partitions(samples, K=K, p=sample_fraction)
        num_iter += 1
        gradients = []
        loss_differences = []
        test_losses = []

        print(f"=== Iteration {num_iter}. alpha={colored(curr_alpha, 'red')} ====")
        for training_data, test_data in tqdm(zip(training_datasets, test_datasets), total=K):
            cov_train = np.cov(training_data, rowvar=False)
            p = cov_train.shape[1]
            train_loss_mle = p + np.log(np.linalg.det(cov_train))
            train_loss_mle_ = compute_loss(cov_train, np.linalg.inv(cov_train))
            if np.isinf(train_loss_mle) or np.isnan(train_loss_mle):
                raise ValueError

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
            test_loss = compute_loss(cov_test, theta)
            dl_dtheta = cov_test - np.linalg.inv(theta)

            alpha_cp.delta = 1
            prob.derivative()
            dtheta_d_alpha = X.delta
            dl_dalpha = np.sum(dl_dtheta * dtheta_d_alpha)

            # # == other way of solving for comparison
            # X.gradient = dl_dtheta
            # prob.backward()
            # dl_dalpha_ = alpha_cp.gradient

            loss_difference = test_loss - train_loss_mle
            loss_differences.append(loss_difference)
            gradients.append(dl_dalpha)
            test_losses.append(test_loss)

            # if abs(dl_dalpha) > 10:
            #     print(loss_difference)
            #     print(dl_dalpha, dl_dalpha_)

        print(f"Average loss difference:", np.mean(loss_differences))
        print(f"Gradient Median: {np.median(gradients)}")
        print(f"Gradient Std: {np.std(gradients)}")
        print(f"Test loss: {np.mean(test_losses)}")
        if cov_heldout is not None:
            print(f"Holdout loss: {compute_loss(cov_heldout, solve_glasso(np.cov(samples, rowvar=False), curr_alpha))}")

        # update alpha
        alpha_path.append(curr_alpha)
        avg_grad = np.median(gradients)
        avg_loss_diff = np.mean(loss_differences)
        new_alpha = gradient_update(curr_alpha, avg_loss_diff, avg_grad, step_num=num_iter)
        new_alpha = max(new_alpha, min_alpha)
        diff = np.linalg.norm(curr_alpha - new_alpha)
        curr_alpha = new_alpha
        test_loss_path.append(np.mean(test_losses))

    # solve again
    theta = solve_glasso(np.cov(samples, rowvar=True), lambda_=curr_alpha)
    return curr_alpha, theta, alpha_path


def solve_glasso(cov, lambda_):
    p = cov.shape[0]
    X = cp.Variable((p, p), PSD=True)
    constraints = [X >> 0]
    objective = cp.Minimize(cp.sum(cp.multiply(cov, X)) - cp.log_det(X) + lambda_*cp.pnorm(X, 1))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    theta = X.value
    return theta


def solve_glasso2(cov, lambda_):
    return graphical_lasso(cov, lambda_)[1]


def compute_loss(cov, theta):
    return np.sum(cov * theta) - np.log(np.linalg.det(theta))


if __name__ == '__main__':
    import causaldag as cd
    from sklearn.covariance import GraphicalLassoCV, graphical_lasso
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set()

    d = cd.rand.directed_erdos(12, .1)
    g = cd.rand.rand_weights(d)
    samples = g.sample(100)
    train_cov = np.cov(samples, rowvar=False)
    train_prec = np.linalg.inv(train_cov)
    heldout_samples = g.sample(1000)
    cov_heldout = np.cov(heldout_samples, rowvar=False)

    # alphas = np.linspace(.01, .6, 12)
    # thetas = [solve_glasso(train_cov, alpha) for alpha in alphas]
    # nll_train = [compute_loss(train_cov, theta) for theta in thetas]
    # nll_heldout = [compute_loss(cov_heldout, theta) for theta in thetas]
    # nll_true = compute_loss(g.covariance, g.precision)
    # plt.axhline(nll_true, color='k', linestyle='--')
    # plt.plot(alphas, nll_train, label="Train")
    # plt.plot(alphas, nll_heldout, label="Heldout")
    #
    # plt.xlabel("Alpha")
    # plt.ylabel("NLL")
    # plt.legend()
    # plt.show()

    print("Running CVGM")
    cvgm_alpha, cvgm_theta, cvgm_path = cvgm_glasso(samples, alpha0=.01, K=20, max_iters=20, cov_heldout=cov_heldout)
    preselected_alphas = [.1, .2, .5, 1, 5, 10]
    thetas_preselected = [solve_glasso(train_cov, lambda_) for lambda_ in preselected_alphas]
    thetas = [solve_glasso(train_cov, lambda_) for lambda_ in cvgm_path]
    nlls = [compute_loss(cov_heldout, theta) for theta in thetas]
    nlls_preselected = [compute_loss(cov_heldout, theta) for theta in thetas_preselected]

    for nll, alpha, color in zip(nlls_preselected, preselected_alphas, sns.color_palette()):
        plt.axhline(y=nll, label=alpha, color=color, linestyle='--')
    plt.plot(nlls)
    plt.xlabel("Step")
    plt.ylabel("Negative Log likelihood")
    plt.legend()
    plt.show()

    # plt.plot(cvgm_path)
    # plt.xlabel("Step")
    # plt.ylabel("Lambda")
    # plt.show()
    # print(f"CVGM alpha: {cvgm_alpha}")

    # print("Running grid search")
    # grid_search_alpha, grid_search_theta = grid_search_cv(samples, np.linspace(.1, 1, num=5))
    # print(f"Grid search alpha: {grid_search_alpha}")
    #
    # p = cov_heldout.shape[0]
    # print(f"CVGM loss: {.5*np.sum(cov_heldout * cvgm_theta) - .5*np.log(np.linalg.det(cvgm_theta)) + p/2 * np.log(2*np.pi)}")
    # print(f"CV loss: {.5*np.sum(cov_heldout * grid_search_theta) - .5*np.log(np.linalg.det(grid_search_theta)) + p/2 * np.log(2*np.pi)}")

    # print(f"Selected alpha from standard CV: {glcv.alpha_}")
    # print(f"Alphas: {glcv.cv_alphas_}")
    # print(f"Average log likelihoods: {glcv.grid_scores_.mean(axis=1)}")
    #
    # glcv_theta = np.linalg.inv(glcv.covariance_)
    #
    # glcv_edges = {frozenset({i, j}) for i, j in zip(*np.nonzero(~np.isclose(glcv_theta, 0))) if i != j}
    # cvgm_edges = {frozenset({i, j}) for i, j in zip(*np.nonzero(~np.isclose(cvgm_theta, 0))) if i != j}
    # print("False negatives in GLCV", len(d.moral_graph().edges - glcv_edges))
    # print("False negatives in CVGM", len(d.moral_graph().edges - cvgm_edges))
    # print("False positives in GLCV", len(glcv_edges - d.moral_graph().edges))
    # print("False positives in CVGM", len(cvgm_edges - d.moral_graph().edges))
