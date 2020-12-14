import causaldag as cd
import numpy as np
from applications.glasso.cvgm import cvgm

num_nodes = 10
nsamples = 100
density = .3
num_graphs = 2
SAVE_SOLUTION = False

dags = cd.rand.directed_erdos(num_nodes, density, size=num_graphs)
gdags = [cd.rand.rand_weights(d) for d in dags]
true_precisions = [g.precision for g in gdags]
samples_list = [g.sample(nsamples) for g in gdags]
sample_dict = dict(enumerate(samples_list))
np.savez(f"applications/glasso/samples_p={num_nodes},n={nsamples},rho={density}.npz", sample_dict)

if SAVE_SOLUTION:
    for graph_ix, samples in enumerate(samples_list):
        prob = None
        selected_lambda, estimated_precision = cvgm(samples, prob, alpha0=1.)


