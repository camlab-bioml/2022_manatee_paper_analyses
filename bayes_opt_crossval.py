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
import citeseq_exp_setup as citeseq_exp

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
def bayes_opt_crossval(experiment, 
                callback, 
                optimise_iter, 
                train_x, 
                train_y, 
                data_train, 
                adata_train, 
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
                data_test,
                adata_test,
                fold,
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
                print(f"Task noises: {task_obs_noises.numpy()}")

                rbf_lscale = model.covar_module.data_covar_module.lengthscale.item()

                corr_means = get_mean_corr(intertask_corr)
                print(f"Corr means: {corr_means.numpy()}")

                is_max, mean_dx, mean_dx2, dx_zeros, min_dx2 = gp_dx.desirable_max(torch.linspace(0, 1, 50), rbf_lscale, task_obs_noises, K_T, train_x, train_y)
                print(f"Desirable max: {is_max.numpy()}")

                lambda_logits, partial_incl_log_probs = acq.logp_include_no_BF_match(task_obs_noises, corr_means, is_max, ablate)
                lambda_probs = torch.exp(lambda_logits)
                partial_lambda = torch.exp(partial_incl_log_probs)
            elif strategy == 'random prob':
                lambda_probs = torch.rand(train_y.shape[1])
                wandb_plot_title = 'random scalarization'
            else:
                print(f"{strategy} is an invalid strategy.")

            acq_params[0] = lambda_probs
            print(f"Lambda probs are: {lambda_probs.numpy()}")
            assert (lambda_probs > 0.).all(), "Some p(lambda_k=1|B_k) is <= 0."

            # Pass in step t
            acq_params[1] = j+1

        # Placeholder for the solution's loss
        sol_loss = float("Inf")

        # Define samples to evalaute acqusition function on
        samples = torch.rand(100, 1)
        # Find initialisation for acqusition function optimisation
        x_probe = acq.sample_to_init_opt(acq_fun, samples, model, acq_params)
        x_probe = torch.logit(x_probe)
        x_probe = x_probe.clone()
        x_probe.requires_grad = True

        if acq_fun == acq.mobo_ucb_scalarized:
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
                f_next, ari, nmi = true_f(x_next_range, data_train, adata_train)            
                log_dict[f"ARI/Fold {fold}/{wandb_plot_title}"] = ari.item()
                log_dict[f"NMI/Fold {fold}/{wandb_plot_title}"] = nmi.item()
                
                _, ari_test, nmi_test = true_f(x_next_range, data_test, adata_test)            
                log_dict[f"ARI_test/Fold {fold}/{wandb_plot_title}"] = ari_test.item()
                log_dict[f"NMI_test/Fold {fold}/{wandb_plot_title}"] = nmi_test.item()

            elif experiment == 'citeseq':
                print(f"Next point in original scale: {x_next_range.item():.3f}")
                f_next, ari, nmi, hvgs = true_f(x_next_range, data_train, adata_train)            
                log_dict[f"ARI/Fold {fold}/{wandb_plot_title}"] = ari.item()
                log_dict[f"NMI/Fold {fold}/{wandb_plot_title}"] = nmi.item()

                ari_test, nmi_test = citeseq_exp.probe_test(x_next_range, data_test, adata_test, hvgs)            
                log_dict[f"ARI_test/Fold {fold}/{wandb_plot_title}"] = ari_test.item()
                log_dict[f"NMI_test/Fold {fold}/{wandb_plot_title}"] = nmi_test.item()

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

            log_dict[f"Solution/Fold {fold}/{wandb_plot_title}"] = x_next_range.item()
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
