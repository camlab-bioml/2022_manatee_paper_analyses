import torch
import gpytorch
import torch.nn.functional as F
import math
import numpy as np
from gpytorch.utils import errors
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import acquis_func as acq
import gp_inference as gp
import gp_derivative as gp_dx
import plotting as plotting
from botorch import fit_gpytorch_model
from botorch.sampling.samplers import SobolQMCNormalSampler
from usemo.model import GaussianProcess
import scipy
from platypus import NSGAII, Problem, Real
import sobol_seq
from pygmo import hypervolume
from usemo.acquisitions import UCB, LCB, TS, ei, pi,compute_beta

# Compute correlation matrix from covariance matrix
def get_corr(B, v):
    intertask_covar = gp.intertask_kernel(B, v)
    intertask_corr = torch.zeros(intertask_covar.shape)
    for i in range(intertask_covar.shape[0]):
        for j in range(intertask_covar.shape[1]):
            intertask_corr[i,j] = intertask_covar[i,j] / (torch.sqrt(intertask_covar[i,i]) * torch.sqrt(intertask_covar[j,j]))
    return intertask_corr

# Compute mean positive correlation across k != k' pairs
def get_mean_corr(corr_matrix):
    assert corr_matrix.shape[0] == corr_matrix.shape[1], "Correlation matrix is not square"
    num_tasks = corr_matrix.shape[0]
    mask = torch.diag(torch.ones(num_tasks))
    corr_means = torch.sum((1. - mask) * F.relu(corr_matrix), axis=1)
    corr_means[corr_means == 0.] = 1e-3
    corr_means = corr_means / (num_tasks-1)
    return corr_means

# Initialise multi-output GP with previous parameters
def load_gp_params_multi(model, raw_task_noises, task_covar_factor, task_raw_var, raw_lscale):
    model.likelihood.raw_task_noises = raw_task_noises
    model.covar_module.task_covar_module.covar_factor = task_covar_factor
    model.covar_module.task_covar_module.raw_var = task_raw_var
    model.covar_module.data_covar_module.raw_lengthscale = raw_lscale
    return model

# Save optimised parameters for multi-output GP
def save_gp_params_multi(model):
    saved_raw_task_noises = model.likelihood.raw_task_noises 
    saved_task_covar_factor = model.covar_module.task_covar_module.covar_factor 
    saved_task_raw_var = model.covar_module.task_covar_module.raw_var 
    saved_raw_lscale = model.covar_module.data_covar_module.raw_lengthscale 
    return saved_raw_task_noises, saved_task_covar_factor, saved_task_raw_var, saved_raw_lscale


# Run Bayesian optimisation
def bayes_opt(experiment, 
                callback, 
                optimise_iter, 
                train_x, 
                train_y, 
                data, 
                adata, 
                likelihood, 
                strategy, 
                model_class, 
                mll_class, 
                acq_fun, 
                acq_params, 
                true_f, 
                x_min, 
                x_max, 
                labels, 
                ablate,
                plot=False, 
                gp_fitting_iter=5):

    num_orig_train_pts = train_x.shape[0]
    num_Bk = 3
    num_tasks = train_y.shape[1]
    acq_f_vals_all = torch.zeros((optimise_iter, 100))
    train_y_original = train_y.detach().clone()
    init_opt_num = 5
    output_dict = {}

    likelihood.train()
    for j in range(optimise_iter):
        print(f"Iteration {j}.")
        if j==0:
            # Center the data 
            # Then, it gets centered after adding a new pt
            train_y = acq.z_score(train_y)
            assert not np.isnan(train_y).any(), "There are NaNs in train_y"

        best_loss = 100.
        tries = 0
        for init_n in range(init_opt_num):
            while True:
                if (tries < 20):
                    try:
                        likelihood.raw_task_noises = torch.nn.Parameter(torch.tensor([0.]).repeat(train_y.shape[1]), requires_grad=True)
                        # New init with current training set
                        model = model_class(train_x, train_y, likelihood, train_y.shape[1])
                        optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')
                        mll = mll_class(likelihood, model)
                        model, curr_losses_gp = gp.fit_gp(model, optimizer, mll, train_x, train_y, gp_fitting_iter)
                        break
                    except (errors.NanError, errors.NotPSDError) as err:
                        print(f"{type(err)} raised, re-initialising...")
                        tries = tries + 1
                else:
                    assert False, f"Raised error when attempting to fit GP {tries} times"

            print(f"Iteration {j}, {init_n}, GP Loss: {curr_losses_gp[-1]:.6f}")

            if curr_losses_gp[-1] < best_loss:
                best_loss = curr_losses_gp[-1]
                # Save the best model's parameters to load at iteration 1
                saved_raw_task_noises, saved_task_covar_factor, saved_task_raw_var, saved_raw_lscale = save_gp_params_multi(model)
            
        # Load the saved parameters from the best initialisation
        model = load_gp_params_multi(model, saved_raw_task_noises, saved_task_covar_factor, saved_task_raw_var, saved_raw_lscale)

        with torch.no_grad():
            log_dict = {}
            log_dict["iteration"] = j
            log_titles = labels
            if strategy == 'manatee':
                wandb_plot_title = 'manatee'
                # Get inter-task covariance matrix
                task_covar_module_var = model.covar_module.task_covar_module.var
                intertask_corr = get_corr(model.covar_module.task_covar_module.covar_factor.detach(), task_covar_module_var.detach())
                K_T = gp.intertask_kernel(model.covar_module.task_covar_module.covar_factor.detach(), task_covar_module_var.detach())
                
                # Get task observation noises
                task_obs_noises = likelihood.task_noises
                for task in range(num_tasks):
                    log_dict[f"Noise/{strategy}/{log_titles[task]}"] = task_obs_noises[task].item()
                print(f"Task noises: {task_obs_noises.numpy()}")

                rbf_lscale = model.covar_module.data_covar_module.lengthscale.item()
                log_dict[f"Lengthscale/{strategy}"] = rbf_lscale

                corr_means = get_mean_corr(intertask_corr)
                print(f"Corr means: {corr_means.numpy()}")
                for task in range(num_tasks):
                    log_dict[f"Correlation/{strategy}/{log_titles[task]}"] = corr_means[task].item()

                is_max, mean_dx, mean_dx2, dx_zeros, min_dx2 = gp_dx.desirable_max(torch.linspace(0, 1, 50), rbf_lscale, task_obs_noises, K_T, train_x, train_y)
                print(f"Desirable max: {is_max.numpy()}")
                for task in range(num_tasks):
                    log_dict[f"Max/{strategy}/{log_titles[task]}"] = is_max[task].item()

                # Compute p(lambda=1|B_k)
                lambda_logits, partial_incl_log_probs = acq.logp_include_no_BF_match(task_obs_noises, corr_means, is_max, ablate)
                lambda_probs = torch.exp(lambda_logits)
                partial_lambda = torch.exp(partial_incl_log_probs)
            elif strategy == 'random prob':
                lambda_probs = torch.rand(train_y.shape[1])
                wandb_plot_title = 'random scalarization'
            else:
                print(f"{strategy} is an invalid strategy.")

            acq_params[0] = lambda_probs
            for task in range(num_tasks):
                log_dict[f"Inclusion probability/{wandb_plot_title}/{log_titles[task]}"] = lambda_probs[task].item()
            if strategy=='manatee':
                for task in range(num_tasks):
                    log_dict[f"Partial inclusion probability/{strategy}/noise/{log_titles[task]}"] = partial_lambda[task,0].item()
                    log_dict[f"Partial inclusion probability/{strategy}/correlation/{log_titles[task]}"] = partial_lambda[task,1].item()
                    log_dict[f"Partial inclusion probability/{strategy}/max/{log_titles[task]}"] = partial_lambda[task,2].item()
            print(f"Lambda probs are: {lambda_probs.numpy()}")
            assert (lambda_probs > 0.).all(), "Some p(lambda_k=1|B_k) is <= 0."

            # Pass in step t
            acq_params[1] = j+1

        # Placeholder for the solution's loss
        sol_loss = float("Inf")

        # Define samples to evalaute acqusition function on
        samples = torch.rand(100, 1)

        if acq_fun == acq.mobo_ucb_scalarized:
            all_lambda_vectors, all_lambda_vector_probs = acq.exhaustive_lambda_vector_probs(lambda_probs)
            if j == 0:
                acq_params.append(all_lambda_vectors)
                acq_params.append(all_lambda_vector_probs)
            else:
                acq_params[2] = all_lambda_vectors
                acq_params[3] = all_lambda_vector_probs

        # Find initialisation for acqusition function optimisation
        x_probe = acq.sample_to_init_opt(acq_fun, samples, model, acq_params)
        x_probe = torch.logit(x_probe)
        x_probe = x_probe.clone()
        x_probe.requires_grad = True

        if acq_fun == acq.mobo_ucb_scalarized:
            optimizer = torch.optim.LBFGS([x_probe], line_search_fn='strong_wolfe')
            scheduler = None

        elif acq_fun == acq.mobo_ucb_scalarized_samples:
            optimizer = torch.optim.Adam([x_probe])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

        elif acq_fun == acq.mobo_acq:
            optimizer = torch.optim.LBFGS([x_probe], line_search_fn='strong_wolfe')
            scheduler = None
        else:
            assert False, f"{acq_fun} is an invalid acquisition function"
        
        # Optimise the acquisition curve
        x_probe, losses_acq = acq.optimise_acq(optimizer, acq_fun, x_probe, model, acq_params, scheduler=scheduler)

        with torch.no_grad():
            # Define test locations for plotting only
            test_x = torch.linspace(0, 1, 100)
            # Compute p(f*|x*, f, X)
            f_pred, observed_pred = gp.post_gp(model, likelihood, test_x)          
            if plot:
                # Plot GP fit to current training set
                plotting.plot_gp_sampled(train_x, train_y, num_orig_train_pts, test_x, f_pred, observed_pred, labels)

            # Add next location to training set
            x_next = acq.bounded_x(x_probe)
            print(f"Next point in (0,1): {x_next.item():.3f}")
            if len(train_x.shape) > 1:
                x_next = torch.reshape(x_next, (x_next.shape[0], 1))
            train_x = torch.cat([train_x, x_next], dim=0)

            # Probe next location, f(x*), at the correct place in original x scale
            x_next_range = x_next * (x_max - x_min) + x_min

            if experiment == 'toy':
                print(f"Next point in original scale: {x_next_range.item():.3f}")
                f_next = true_f(x_next_range)

            elif experiment == 'imc':
                print(f"Next point in original scale: {x_next_range.item():.3f}")
                f_next, ari, nmi = true_f(x_next_range, data, adata)            
                log_dict[f"ARI/{wandb_plot_title}"] = ari.item()
                log_dict[f"NMI/{wandb_plot_title}"] = nmi.item()

            elif experiment == 'citeseq':
                print(f"Next point in original scale: {x_next_range.item():.3f}")
                f_next, ari, nmi, _ = true_f(x_next_range, data, adata)            
                log_dict[f"ARI/{wandb_plot_title}"] = ari.item()
                log_dict[f"NMI/{wandb_plot_title}"] = nmi.item()

            else: 
                print(f"{experiment} is an invalid experiment type.")

            # Add sampled point to original unscaled dataset
            train_y_original = torch.cat([train_y_original, f_next], dim=0)
            # Standardise the new current training set
            train_y = acq.z_score(train_y_original)

            if plot:
                # Compute acquisition function values for plotting
                acq_f_vals = torch.zeros(test_x.shape[0])
                for i in range(test_x.shape[0]):
                    acq_f_vals[i] = acq_fun(test_x[i, None], model, acq_params)
                acq_f_vals_all[j,:] = acq_f_vals

                # Plot acquisition function
                plotting.plot_acq(test_x, acq_f_vals, x_next, xlabel='x', ylabel='a(x)', 
                    title=f"{j}: acquisition f")

            log_dict[f"Solution/{wandb_plot_title}"] = x_next_range.item()
            callback(log_dict)
            
            output_dict[f"model/{strategy}/iter {j}"] = model
            output_dict[f"likelihood/{strategy}/iter {j}"] = likelihood
            if strategy == 'manatee':
                output_dict[f"mean_dx2/{strategy}/iter {j}"] = mean_dx2
                output_dict[f"mean_dx/{strategy}/iter {j}"] = mean_dx
                output_dict[f"intertask_corr/{strategy}/iter {j}"] = intertask_corr
         
    output_dict[f"train_x/{strategy}"] = train_x
    output_dict[f"train_y/{strategy}"] = train_y
    output_dict[f"train_y_original/{strategy}"] = train_y_original
    if plot:
        output_dict[f"acq_f_vals_all/{strategy}"] = acq_f_vals_all
            
    return output_dict

# Run Bayesian optimisation
def bayes_opt_random(experiment, 
                        callback, 
                        optimise_iter, 
                        train_x, 
                        train_y, 
                        data, 
                        adata, 
                        true_f, 
                        x_min, 
                        x_max, 
                        labels, 
                        plot=False, 
                        gp_fitting_iter=5):

    num_orig_train_pts = train_x.shape[0]
    train_y_original = train_y.detach().clone()
    output_dict = {}

    for j in range(optimise_iter):
        print(f"Iteration {j}.")
        if j==0:
            # Center the data
            # Then, it gets centered after adding a new pt
            train_y = acq.z_score(train_y)
            assert not np.isnan(train_y).any(), "There are NaNs in train_y"

        with torch.no_grad():
            log_dict = {}
            log_dict["iteration"] = j
            if plot:
                # Define test locations for plotting only
                test_x = torch.linspace(0, 1, 100)
                # Plot GP fit to current training set
                plotting.plot_data_sampled(train_x, train_y, num_orig_train_pts, labels)

            # Add next location to training set
            x_next = torch.rand(1)
            print(f"Next point in (0,1): {x_next.item():.3f}")
            if len(train_x.shape) > 1:
                x_next = torch.reshape(x_next, (x_next.shape[0], 1))
            train_x = torch.cat([train_x, x_next], dim=0)

            # Probe next location, f(x*), at the correct place in original x scale
            x_next_range = x_next * (x_max - x_min) + x_min

            if experiment == 'toy':
                print(f"Next point in original scale: {x_next_range.item():.3f}")
                f_next = true_f(x_next_range)

            elif experiment == 'imc':
                print(f"Next point in original scale: {x_next_range.item():.3f}")
                f_next, ari, nmi = true_f(x_next_range, data, adata)
                log_dict["ARI/random acquisition"] = ari.item()
                log_dict["NMI/random acquisition"] = nmi.item()

            elif experiment == 'citeseq':
                print(f"Next point in original scale: {x_next_range.item():.3f}")
                f_next, ari, nmi, _ = true_f(x_next_range, data, adata)
                log_dict["ARI/random acquisition"] = ari.item()
                log_dict["NMI/random acquisition"] = nmi.item()

            else:
                print(f"{experiment} is an invalid experiment type.")

            # Add sampled point to original unscaled dataset
            train_y_original = torch.cat([train_y_original, f_next], dim=0)
            # Standardise the new current training set
            train_y = acq.z_score(train_y_original)

            log_dict["Solution/random acquisition"] = x_next_range.item()
            callback(log_dict)

    output_dict["train_x/random loc"] = train_x
    output_dict["train_y/random loc"] = train_y
    output_dict["train_y_original/random loc"] = train_y_original
             
    return output_dict

MC_SAMPLES = 128

# Run botorch
def bayes_opt_botorch(experiment, 
                        callback, 
                        optimise_iter, 
                        train_x, 
                        train_y, 
                        data, 
                        adata, 
                        true_f, 
                        x_min, 
                        x_max, 
                        labels, 
                        plot=False, 
                        gp_fitting_iter=5):

    num_orig_train_pts = train_x.shape[0]
    train_y_original = train_y.detach().clone()
    num_tasks = train_y.shape[1]
    output_dict = {}

    for j in range(optimise_iter):
        print(f"Iteration {j}.")

        if j == 0:
            ref_point = torch.min(train_y, axis=0).values 
            standard_bounds = torch.tensor([[0.], [1.]])

        train_x_qnehvi = train_x.detach().clone()
        train_obj_qnehvi = train_y_original

        # Init botorch model
        mll_qnehvi, model_qnehvi = gp.initialize_botorch_model(train_x_qnehvi, train_obj_qnehvi, standard_bounds)
        print("Initialised model.")

        try:
            # fit the model
            fit_gpytorch_model(mll_qnehvi, max_retries=20)
            print("Fitted.")

            # define the qEI and qNEI acquisition modules using a QMC sampler
            batch_range = (0, -1)
            qnehvi_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES, batch_range=batch_range)

            # optimize acquisition functions and get new observations
            
            print("Optimizing...")
            new_x_qnehvi = gp.optimize_qnehvi_and_get_observation(
                model_qnehvi, train_x_qnehvi, train_obj_qnehvi, qnehvi_sampler, ref_point, standard_bounds 
            )
            print(f"iteration {j}: new point: {new_x_qnehvi}")

            with torch.no_grad():
                log_dict = {}
                log_dict["iteration"] = j

                # Add next location to training set
                train_x = torch.cat([train_x, new_x_qnehvi])

                # Probe next location, f(x*), at the correct place in original x scale
                x_next_range = new_x_qnehvi * (x_max - x_min) + x_min

                if experiment == 'toy':
                    print(f"Next point in original scale: {x_next_range.item():.3f}")
                    f_next = true_f(x_next_range)

                elif experiment == 'imc':
                    print(f"Next point in original scale: {x_next_range.item():.3f}")
                    f_next, ari, nmi = true_f(x_next_range, data, adata)
                    log_dict["ARI/botorch"] = ari.item()
                    log_dict["NMI/botorch"] = nmi.item()

                elif experiment == 'citeseq':
                    print(f"Next point in original scale: {x_next_range.item():.3f}")
                    f_next, ari, nmi, _ = true_f(x_next_range, data, adata)
                    log_dict["ARI/botorch"] = ari.item()
                    log_dict["NMI/botorch"] = nmi.item()

                else:
                    print(f"{experiment} is an invalid experiment type.")

                # Add sampled point to original unscaled dataset
                train_y_original = torch.cat([train_y_original, f_next])

                log_dict["Solution/botorch"] = x_next_range.item()
                callback(log_dict)

        except RuntimeError as e:
            print(f"botorch failed with a RuntimeError\n")
            print(f"Error: {e}")
            break

    with torch.no_grad():
        # At the end, save sampled datasets
        output_dict["train_x/botorch"] = train_x
        output_dict["train_y/botorch"] = train_y
        output_dict["train_y_original/botorch"] = train_y_original
            
    return output_dict
    
# Run botorch
def bayes_opt_qparego(experiment, 
                        callback, 
                        optimise_iter, 
                        train_x, 
                        train_y, 
                        data, 
                        adata, 
                        true_f, 
                        x_min, 
                        x_max, 
                        labels, 
                        plot=False, 
                        gp_fitting_iter=5):

    num_orig_train_pts = train_x.shape[0]
    train_y_original = train_y.detach().clone()
    num_tasks = train_y.shape[1]
    output_dict = {}

    for j in range(optimise_iter):
        print(f"Iteration {j}.")

        if j == 0:
            standard_bounds = torch.tensor([[0.], [1.]])

        train_x_qparego = train_x.detach().clone()
        train_obj_qparego = train_y_original

        # Init botorch model
        mll_qparego, model_qparego = gp.initialize_botorch_model(train_x_qparego, train_obj_qparego, standard_bounds)
        print("Initialised model.")

        try:
            # fit the model
            fit_gpytorch_model(mll_qparego, max_retries=20)
            print("Fitted.")

            # define the qEI and qNEI acquisition modules using a QMC sampler
            batch_range = (0, -1)
            qparego_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES, batch_range=batch_range)

            # optimize acquisition functions and get new observations
            
            print("Optimizing...")
            new_x_qparego = gp.optimize_qnparego_and_get_observation(
                model_qparego, train_x_qparego, train_obj_qparego, qparego_sampler, standard_bounds 
            )
            print(f"iteration {j}: new point: {new_x_qparego}")

            with torch.no_grad():
                log_dict = {}
                log_dict["iteration"] = j

                # Add next location to training set
                train_x = torch.cat([train_x, new_x_qparego])

                # Probe next location, f(x*), at the correct place in original x scale
                x_next_range = new_x_qparego * (x_max - x_min) + x_min

                if experiment == 'toy':
                    print(f"Next point in original scale: {x_next_range.item():.3f}")
                    f_next = true_f(x_next_range)

                elif experiment == 'imc':
                    print(f"Next point in original scale: {x_next_range.item():.3f}")
                    f_next, ari, nmi = true_f(x_next_range, data, adata)
                    log_dict["ARI/qparego"] = ari.item()
                    log_dict["NMI/qparego"] = nmi.item()

                elif experiment == 'citeseq':
                    print(f"Next point in original scale: {x_next_range.item():.3f}")
                    f_next, ari, nmi, _ = true_f(x_next_range, data, adata)
                    log_dict["ARI/qparego"] = ari.item()
                    log_dict["NMI/qparego"] = nmi.item()

                else:
                    print(f"{experiment} is an invalid experiment type.")

                # Add sampled point to original unscaled dataset
                train_y_original = torch.cat([train_y_original, f_next])

                log_dict["Solution/qparego"] = x_next_range.item()
                callback(log_dict)

        except RuntimeError as e:
            print(f"qparego failed with a RuntimeError\n")
            print(f"Error: {e}")
            break
            
    with torch.no_grad():
        # At the end, save sampled datasets
        output_dict["train_x/qparego"] = train_x
        output_dict["train_y/qparego"] = train_y
        output_dict["train_y_original/qparego"] = train_y_original
            
    return output_dict

# Run usemo
# Adapted from https://github.com/belakaria/USeMO/blob/master/main.py
def bayes_opt_usemo(experiment, 
                        callback, 
                        optimise_iter, 
                        train_x, 
                        train_y, 
                        data, 
                        adata, 
                        true_f, 
                        x_min, 
                        x_max):

    num_orig_train_pts = train_x.shape[0]
    train_y_original = train_y.detach().clone()
    num_tasks = train_y.shape[1]
    output_dict = {}

    referencePoint = [1e5]*num_tasks
    bound=[0,1] 
    d = train_x.shape[1]
    Fun_bounds = [bound]*d
    grid = sobol_seq.i4_sobol_generate(d,1000,np.random.randint(0,1000))
    design_index = np.random.randint(0, grid.shape[0])

    acquisation=TS
    batch_size=1

    GPs=[]
    for i in range(num_tasks):
        GPs.append(GaussianProcess(d))

    for j in range(optimise_iter):
        print(f"Iteration {j}.")

        train_x_usemo = np.array(train_x.detach().clone())
        train_y_usemo = np.array(train_y_original)

        if j == 0:
            for k in range(num_orig_train_pts):
                for i in range(num_tasks):
                    GPs[i].addSample(train_x_usemo[k,:], -train_y_usemo[k,i])
        else:
            for i in range(num_tasks):
                GPs[i].addSample(train_x_usemo[-1,:], -train_y_usemo[-1,i])

        for i in range(num_tasks):   
            GPs[i].fitModel()

        beta=compute_beta(j+1,d)
        cheap_pareto_set=[]

        def CMO(x):
            x=np.asarray(x)
            return [acquisation(x,beta,GPs[i])[0] for i in range(len(GPs))]

        problem = Problem(d, num_tasks)
        problem.types[:] = Real(bound[0], bound[1])
        problem.function = CMO
        algorithm = NSGAII(problem)
        algorithm.run(2500)
        cheap_pareto_set=[solution.variables for solution in algorithm.result]
        cheap_pareto_set_unique=[]
        for i in range(len(cheap_pareto_set)):
            if (any((cheap_pareto_set[i] == x).all() for x in GPs[0].xValues))==False:
                cheap_pareto_set_unique.append(cheap_pareto_set[i])

        UBs=[[GPs[i].getPrediction(np.asarray(np.asarray(x)))[0][0]+beta*GPs[i].getPrediction(np.asarray(np.asarray(x)))[1][0] for i in range(len(GPs))] for x in cheap_pareto_set_unique]
        LBs=[[GPs[i].getPrediction(np.asarray(np.asarray(x)))[0][0]-beta*GPs[i].getPrediction(np.asarray(np.asarray(x)))[1][0] for i in range(len(GPs))] for x in cheap_pareto_set_unique]
        uncertaities= [scipy.spatial.Rectangle(UBs[i], LBs[i]).volume() for i in range(len(cheap_pareto_set_unique))]

        batch_indecies=np.argsort(uncertaities)[::-1][:batch_size]
        batch=[cheap_pareto_set_unique[i] for i in batch_indecies]

        for x_best in batch:
           new_x_usemo = torch.tensor([x_best]) 

        print(f"iteration {j}: new point: {new_x_usemo}")

        with torch.no_grad():
            log_dict = {}
            log_dict["iteration"] = j

            # Add next location to training set
            train_x = torch.cat([train_x, new_x_usemo])

            # Probe next location, f(x*), at the correct place in original x scale
            x_next_range = new_x_usemo * (x_max - x_min) + x_min

            if experiment == 'toy':
                print(f"Next point in original scale: {x_next_range.item():.3f}")
                f_next = true_f(x_next_range)

            elif experiment == 'sklearn':
                print(f"Next point in original scale: {x_next_range.item():.3f}")
                f_next, ari, nmi = true_f(x_next_range, data, adata.copy())
                log_dict["ARI/usemo"] = ari.item()
                log_dict["NMI/usemo"] = nmi.item()

            elif experiment == 'imc':
                print(f"Next point in original scale: {x_next_range.item():.3f}")
                f_next, ari, nmi = true_f(x_next_range, data, adata)
                log_dict["ARI/usemo"] = ari.item()
                log_dict["NMI/usemo"] = nmi.item()

            elif experiment == 'citeseq':
                print(f"Next point in original scale: {x_next_range.item():.3f}")
                f_next, ari, nmi, _ = true_f(x_next_range, data, adata)
                log_dict["ARI/usemo"] = ari.item()
                log_dict["NMI/usemo"] = nmi.item()

            else:
                print(f"{experiment} is an invalid experiment type.")

            # Add sampled point to original unscaled dataset
            train_y_original = torch.cat([train_y_original, f_next])

            log_dict["Solution/usemo"] = x_next_range.item()
            callback(log_dict)

    with torch.no_grad():
        # At the end, save sampled datasets
        output_dict["train_x/usemo"] = train_x
        output_dict["train_y/usemo"] = train_y
        output_dict["train_y_original/usemo"] = train_y_original
        output_dict["model/usemo"] = GPs

    return output_dict
