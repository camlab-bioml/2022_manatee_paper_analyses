import torch
import gpytorch
import numpy as np
from scipy.stats import linregress
from matplotlib import pyplot as plt
plt.style.use("ggplot")

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.GreaterThan(0.1)), num_tasks=num_tasks, rank=num_tasks
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def fit_gp(model, optimizer, mll, train_x, train_y, training_iter=5):
    model.train()
    
    losses_gp = []
    for i in range(training_iter):
        class Closure():
            def __init__(self):
                self.losses = []
            def __call__(self):
                optimizer.zero_grad()
                output = model(train_x)
                self.loss = -mll(output, train_y)
                self.losses.append(self.loss.item())
                self.loss.backward()
                return self.loss
        closure = Closure()
        optimizer.step(closure)
        losses_gp += closure.losses
    return model, losses_gp

def post_gp(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        f_pred = model(test_x)            
        observed_pred = likelihood(f_pred)
    return f_pred, observed_pred

def intertask_kernel(B, v):
    return torch.matmul(B, B.T) + torch.diag(v)

from botorch.models.transforms.outcome import Standardize
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.utils.transforms import unnormalize, normalize
from gpytorch import ExactMarginalLogLikelihood
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.utils import get_default_partitioning_alpha
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.optim.optimize import optimize_acqf_list

def initialize_botorch_model(train_x, train_obj, standard_bounds):
    # define models for objective and constraint
    train_x = normalize(train_x, standard_bounds)
    model = KroneckerMultiTaskGP(
            train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1])
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

BATCH_SIZE = 1
NUM_RESTARTS = 20
RAW_SAMPLES = 1024

def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler, ref_point, standard_bounds):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    alpha = get_default_partitioning_alpha(train_obj.shape[1])
    # partition non-dominated space into disjoint rectangles
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),  # use known reference point
        X_baseline=normalize(train_x, standard_bounds),
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        alpha=alpha,
        sampler=sampler,
        cache_root=False,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=standard_bounds)
    return new_x

def optimize_qnparego_and_get_observation(model, train_x, train_obj, sampler, standard_bounds):
    """Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization
    of the qNParEGO acquisition function, and returns a new candidate and observation."""
    train_x = normalize(train_x, standard_bounds)
    with torch.no_grad():
        pred = model.posterior(train_x).mean
    acq_func_list = []
    num_tasks = train_obj.shape[1]
    for _ in range(BATCH_SIZE):
        weights = sample_simplex(num_tasks).squeeze()
        objective = GenericMCObjective(
            get_chebyshev_scalarization(weights=weights, Y=pred)
        )
        acq_func = qNoisyExpectedImprovement(  # pyre-ignore: [28]
            model=model,
            objective=objective,
            X_baseline=train_x,
            sampler=sampler,
            prune_baseline=True,
        )
        acq_func_list.append(acq_func)
    # optimize
    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_func_list,
        bounds=standard_bounds,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=standard_bounds)
    return new_x
