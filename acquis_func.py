import math
import torch
import torch.nn.functional as F 
import gpytorch
from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.beta import Beta
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def get_ucb_beta(t):
    beta = .125 * torch.tensor(math.log(2*t + 1))
    # This is for numerical safety so sqrt(0) or of values close to 0
    # doesn't break the gradients optimising the acq function
    beta = F.relu(beta) + 1e-6
    return beta

def ucb_acq(x_probe, model, acq_params):
    t = acq_params[0]
    beta = get_ucb_beta(t)
    model.eval()
    # p(f*|x*, f, X)
    f_pred = model(x_probe)
    return f_pred.mean +  torch.sqrt(f_pred.variance * beta)

def scalar_f_linear(acq_f_vals, lambdas):
    return torch.sum(acq_f_vals * lambdas, axis=1)

def mobo_acq(x_probe, model, acq_params):
    lambdas = acq_params[0]
    t = acq_params[1]
    return scalar_f_linear(ucb_acq(x_probe, model, [t]), lambdas = lambdas)

def mc_sample_std(f_pred, p_lambdas, S):
    num_tasks = len(p_lambdas)
    estimates = torch.zeros(S)
    for s in range(S):
        lambda_sample = torch.bernoulli(p_lambdas.clone().detach())
        var_term = torch.sum(lambda_sample**2 * f_pred.variance, axis=1)
        
        covar_indices = torch.triu_indices(num_tasks, num_tasks, offset=1)
        pair_indices = [[covar_indices[0][i].item(), covar_indices[1][i].item()] for i in range(len(covar_indices[0]))]
        lambda_matrix = torch.zeros(num_tasks, num_tasks)
        lambda_pairs = torch.tensor([torch.prod(lambda_sample[i]).item() for i in pair_indices])
        lambda_matrix[covar_indices[0], covar_indices[1]] = lambda_pairs
        covar_term = torch.sum(lambda_matrix * torch.triu(f_pred.covariance_matrix, diagonal=1))
        
        estimated_sum = var_term + 2*covar_term
        assert estimated_sum > -1e-4, f"covariance estimate in acq function is negative and too large: {sum_term}"
        estimates[s] = torch.sqrt(torch.clamp(estimated_sum, 1e-6, 1e10))
    return torch.mean(estimates)

def exhaustive_lambda_vector_probs(p_lambdas):
    num_tasks = p_lambdas.shape[0]
    all_lambda_vectors = torch.cartesian_prod(*[torch.tensor([0.,1.]) for i in range(num_tasks)])
    one_minus_p_lambdas = 1.-p_lambdas
    arr1 = all_lambda_vectors * p_lambdas.view(1, num_tasks)
    arr2 = (1.-all_lambda_vectors) * one_minus_p_lambdas.view(1, num_tasks)
    # these are log probabilities
    all_lambda_vector_probs = torch.sum(torch.log(arr1+arr2), axis=1)
    return all_lambda_vectors, all_lambda_vector_probs

def exhaustive_std(f_pred, all_lambda_vectors, all_lambda_vector_probs):
    num_tasks = all_lambda_vectors.shape[1]
    expectations = torch.zeros(all_lambda_vectors.shape[0])

    for i, vec in enumerate(all_lambda_vectors):
        var_term = torch.sum(vec**2 * f_pred.variance, axis=1)
        
        covar_indices = torch.triu_indices(num_tasks, num_tasks, offset=1)
        pair_indices = [[covar_indices[0][i].item(), covar_indices[1][i].item()] for i in range(len(covar_indices[0]))]
        lambda_matrix = torch.zeros(num_tasks, num_tasks)
        lambda_pairs = torch.tensor([torch.prod(vec[i]).item() for i in pair_indices])
        lambda_matrix[covar_indices[0], covar_indices[1]] = lambda_pairs
        covar_term = torch.sum(lambda_matrix * torch.triu(f_pred.covariance_matrix, diagonal=1))
        
        sum_term = var_term + 2*covar_term
        assert sum_term > -1e-4, f"covariance estimate in acq function is negative and too large: {sum_term}"
        # clamp to prevent backward pass sqrt issues with 0 or small values
        expectations[i] = torch.sqrt(torch.clamp(sum_term, 1e-6, 1e10))

    expectation_terms = torch.exp(all_lambda_vector_probs) * expectations
    return torch.sum(expectation_terms)

def mobo_ucb_scalarized(x_probe, model, acq_params):
    t = acq_params[1]
    lambdas = acq_params[0]
    all_lambda_vectors = acq_params[2]
    all_lambda_vector_probs = acq_params[3]
    beta = get_ucb_beta(t)
    model.eval()
    # p(f*|x*, f, X)
    f_pred = model(x_probe)
    mean = torch.sum(f_pred.mean * lambdas, axis=1)
    std = exhaustive_std(f_pred, all_lambda_vectors, all_lambda_vector_probs) 
    return mean + torch.sqrt(beta) * std

def mobo_ucb_scalarized_samples(x_probe, model, acq_params, S=500):
    t = acq_params[1]
    lambdas = acq_params[0]
    beta = get_ucb_beta(t)
    model.eval()
    # p(f*|x*, f, X)
    f_pred = model(x_probe)
    mean = torch.sum(f_pred.mean * lambdas, axis=1)
    std = mc_sample_std(f_pred, lambdas, S) 
    return mean + torch.sqrt(beta) * std

# Uniform Bernoulli prior on lambda
def lambda_prior(prob=torch.tensor([0.5])):
    return Bernoulli(prob.clone().detach())

def noise_include_no_BF_match(noise):
   noise[noise >= 1.] = 1.-1e-3
   return 2.-2.*noise 

def noise_exclude_no_BF_match(noise):
    return 2.*noise

def objagree_include_no_BF_match(eta):
    return 2.*eta

def objagree_exclude_no_BF_match(eta):
    eta[eta == 1.] = 1.-1e-3
    return 2.-2.*eta

def max_include_bernoulli(p_1=torch.tensor([.75])):
    assert p_1 >= 0. and p_1 <= 1., f'p_1 should be in [0,1], not {p_1}'
    return Bernoulli(p_1)

def max_exclude_bernoulli(p_0=torch.tensor([.25])):
    assert p_0 >= 0. and p_0 <= 1., f'p_0 should be in [0,1], not {p_0}'
    return Bernoulli(p_0)

def logp_include_no_BF_match(obs_noises, corr_means, is_max, ablate):    

    logp_B_include = lambda_prior().log_prob(torch.tensor(1.)) +\
                    torch.log(noise_include_no_BF_match(obs_noises)) +\
                    torch.log(objagree_include_no_BF_match(corr_means)) +\
                    max_include_bernoulli().log_prob(is_max)
    logp_B_exclude = lambda_prior().log_prob(torch.tensor(0.)) +\
                    torch.log(noise_exclude_no_BF_match(obs_noises)) +\
                    torch.log(objagree_exclude_no_BF_match(corr_means)) +\
                    max_exclude_bernoulli().log_prob(is_max)

    if ablate == 'cor':
        logp_B_include = logp_B_include - torch.log(objagree_include_no_BF_match(corr_means)) 
        logp_B_exclude = logp_B_exclude - torch.log(objagree_exclude_no_BF_match(corr_means)) 
    
    elif ablate == 'noise':
        logp_B_include = logp_B_include - torch.log(noise_include_no_BF_match(obs_noises)) 
        logp_B_exclude = logp_B_exclude - torch.log(noise_exclude_no_BF_match(obs_noises)) 

    elif ablate == 'max':
        logp_B_include = logp_B_include - max_include_bernoulli().log_prob(is_max)
        logp_B_exclude = logp_B_exclude - max_exclude_bernoulli().log_prob(is_max)

    elif ablate != 'none':
        assert False, f"{ablate} is not a valid behaviour to ablate. Possible options: cor, noise, max, none"

    p_B = torch.exp(logp_B_include) + torch.exp(logp_B_exclude)

    num_Bk = 3
    partial_incl_log_probs = torch.zeros((logp_B_include.shape[0], num_Bk))

    log_noise_lambda1 = lambda_prior().log_prob(torch.tensor(1.)) +\
                    torch.log(noise_include_no_BF_match(obs_noises))

    log_corr_lambda1 = lambda_prior().log_prob(torch.tensor(1.)) +\
                    torch.log(objagree_include_no_BF_match(corr_means))

    log_max_lambda1 = lambda_prior().log_prob(torch.tensor(1.)) +\
                    max_include_bernoulli().log_prob(is_max) 

    log_noise_lambda0 = lambda_prior().log_prob(torch.tensor(0.)) +\
                    torch.log(noise_exclude_no_BF_match(obs_noises))

    log_corr_lambda0 = lambda_prior().log_prob(torch.tensor(0.)) +\
                    torch.log(objagree_exclude_no_BF_match(corr_means))

    log_max_lambda0 = lambda_prior().log_prob(torch.tensor(0.)) +\
                    max_exclude_bernoulli().log_prob(is_max) 

    p_noise = torch.exp(log_noise_lambda1) + torch.exp(log_noise_lambda0)
    p_corr = torch.exp(log_corr_lambda1) + torch.exp(log_corr_lambda0)
    p_max = torch.exp(log_max_lambda1) + torch.exp(log_max_lambda0)

    partial_incl_log_probs[:,0] = log_noise_lambda1 - torch.log(p_noise)
    partial_incl_log_probs[:,1] = log_corr_lambda1 - torch.log(p_corr)
    partial_incl_log_probs[:,2] = log_max_lambda1 - torch.log(p_max)

    return logp_B_include - torch.log(p_B), partial_incl_log_probs 

def z_score(y):
    return (y - torch.mean(y, axis=0)) / torch.sqrt(torch.var(y, axis=0))

def bounded_x(x, min_x=0, max_x=1):
    return (max_x - min_x) * torch.sigmoid(x) + min_x

def optimise_acq(optimizer, acq_f, x_probe, model, acq_params, scheduler):
    if acq_f == mobo_ucb_scalarized:
        x_probe, losses_acq = optimise_acq_lbfgs(optimizer, acq_f, x_probe, model, acq_params, scheduler)
    if acq_f == mobo_ucb_scalarized_samples:
        x_probe, losses_acq = optimise_acq_adam(optimizer, acq_f, x_probe, model, acq_params, scheduler)
    if acq_f == mobo_acq:
        x_probe, losses_acq = optimise_acq_lbfgs(optimizer, acq_f, x_probe, model, acq_params, scheduler)
    return x_probe, losses_acq

def optimise_acq_lbfgs(optimizer, acq_f, x_probe, model, acq_params, scheduler, steps=5):
    losses_all = []
    
    assert scheduler is None, "scheduler for LBFGS should be none"
    for i in range(steps):
        class Closure():
            def __init__(self):
                self.losses = []
            def __call__(self):
                optimizer.zero_grad()
                # loss is the acquisition function
                self.loss = -acq_f(bounded_x(x_probe), model, acq_params)
                self.losses.append(self.loss.item())
                self.loss.backward()
                return self.loss
        closure = Closure()
        # Take step
        optimizer.step(closure)
        losses_all += closure.losses
    
    return x_probe, losses_all

def optimise_acq_adam(optimizer, acq_f, x_probe, model, acq_params, scheduler, steps=300):
    losses = []
    assert scheduler.T_max == steps, f"number of steps for scheduler should equal {steps}, not {scheduler.T_max}" 

    for i in range(steps):
        optimizer.zero_grad()
        # loss is the acquisition function
        loss = -acq_f(bounded_x(x_probe), model, acq_params)
        losses.append(loss.item())
        loss.backward()
        # Take step
        optimizer.step()
        scheduler.step()
        if (i % 50) == 0.:
            print(i, loss.item())
        
    return x_probe, losses

def sample_to_init_opt(acq_f, x_probes, model, acq_params):
    
    # Sample points and evaluate acq func to pick init for optimisation
    # Note that because points are sampled uniformly on (0,1) (for testing),
    # no need to pass them through `bounded_x`
    with torch.no_grad():
        max_guess = torch.tensor([-1e6])
        for sample in x_probes:
            if acq_f == mobo_ucb_scalarized:
                acq_f_val = acq_f(sample, model, acq_params)
            elif acq_f == mobo_ucb_scalarized_samples:
                acq_f_val = acq_f(sample, model, acq_params, S=10)
            else:
                acq_f_val = acq_f(sample, model, acq_params)
            if acq_f_val > max_guess:
                max_guess = acq_f_val
                x_probe = sample
        if max_guess == torch.tensor([-1e6]):
            raise ValueError("Sampled acquisition function values are negative.")
        print(f"Initialized from {x_probe}")
    return x_probe
