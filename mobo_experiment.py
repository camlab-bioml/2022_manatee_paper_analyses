import numpy as np
import wandb
import torch
torch.use_deterministic_algorithms(True)
import gpytorch
import random
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import acquis_func as acq
import gp_inference as gp
import bayes_opt as bo
import bayes_opt_crossval as bo_cv
import CallbackClass as Clb
import toy_exp_setup as toy_setup
import imc_exp_setup as imc_setup
import citeseq_exp_setup as citeseq_setup
import pathlib

# Code by kagronick (StackOverflow)
def get_git_revision():
    base_path = '/home/campbell/aselega/Projects/manatee-new'
    git_dir = pathlib.Path(base_path) / '.git'
    with (git_dir / 'HEAD').open('r') as head:
        ref = head.readline().split(' ')[-1].strip()

    with (git_dir / ref).open('r') as git_hash:
        return git_hash.readline().strip()

# This function sets the seed.
def set_seed(seed):
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(a=seed)

# This function sets logging settings (wandb or CallbackClass to be run in notebook or saved to disk)
# and calls `main()`.  
def log_settings(experiment, logging, strategy, x_min, x_max, num_train_pts, optimise_iter, plot, seed, desc, crossval, ablate, ucb_scal):

    from joblib import Memory
    from pathlib import Path
    cachedir = Path(f'~/cache_' + experiment)
    cachedir.mkdir(exist_ok=True, parents=True)
    memory = Memory(cachedir, verbose=0)

    if experiment == "toy":
        data1, data2 = x_min, x_min
    elif experiment == "imc":
        data1, data2 = memory.cache(imc_setup.load_data)()
    elif experiment == "citeseq":
        data1, data2 = memory.cache(citeseq_setup.load_data)()

    set_seed(int(seed))

    labels = get_labels(experiment)
    log_titles = labels

    if logging == 'wandb':
        config = dict(experiment=experiment, logging=logging, x_min=x_min, x_max=x_max, 
                num_train_pts=num_train_pts, optimise_iter=optimise_iter, 
                strategy=strategy,
                seed=seed, commit=get_git_revision(),
                desc=desc)

        if experiment == "toy":
            project_name = "toy-mobo"
        elif experiment == "imc":
            project_name = "imc-mobo"
        elif experiment == 'citeseq':
            project_name = 'citeseq-mobo'

        wandb.init(
              # Set the project where this run will be logged
              project = project_name, 
              # Track hyperparameters and run metadata
              config=config)
        callback = wandb.log
    elif logging == "nolog":
        callback = Clb.CallbackClass(strategy, log_titles)

    # Get returned variables from main() as a dict
    success = False
    if experiment == "imc":
        if crossval: 
            dict_list_all_folds = []
            success_flags_all_folds = []
            for fold in range(5):        
                # Split
                data1_train, data2_train, data1_test, data2_test = imc_setup.split_data(data1, data2)
                mobo_output_dict, success = main_crossval(data1_train, data2_train, data1_test, data2_test, fold, experiment, callback, strategy, x_min, x_max, num_train_pts, optimise_iter, plot, ablate, ucb_scal)
                dict_list_all_folds.append(mobo_output_dict)
                success_flags_all_folds.append(success)

            mobo_output_dict = {k:v for d in dict_list_all_folds for k,v in d.items()} 
            if np.all(success_flags_all_folds):
                success = True
            else:
                success = False
        else:
            mobo_output_dict, success = main(data1, data2, experiment, callback, strategy, x_min, x_max, num_train_pts, optimise_iter, plot, ablate, ucb_scal)

    elif experiment == "citeseq":
        if crossval:
            dict_list_all_folds = []
            success_flags_all_folds = []
            for fold in range(5):
                # Split
                data1_train, data2_train, data1_test, data2_test = citeseq_setup.split_data(data1, data2)
                mobo_output_dict, success = main_crossval(data1_train, data2_train, data1_test, data2_test, fold, experiment, callback, strategy, x_min, x_max, num_train_pts, optimise_iter, plot, ablate, ucb_scal)
                dict_list_all_folds.append(mobo_output_dict)
                success_flags_all_folds.append(success)

            mobo_output_dict = {k:v for d in dict_list_all_folds for k,v in d.items()} 
            if np.all(success_flags_all_folds):
                success = True
            else:
                success = False
        else: 
            mobo_output_dict, success = main(data1, data2, experiment, callback, strategy, x_min, x_max, num_train_pts, optimise_iter, plot, ablate, ucb_scal)

    else:
        mobo_output_dict, success = main(data1, data2, experiment, callback, strategy, x_min, x_max, num_train_pts, optimise_iter, plot, ablate, ucb_scal)
    
    if logging == 'wandb':
        wandb.log({'botorch_succeeded':success})
        wandb.finish()

def get_labels(experiment):
    # Get labels based on experiment
    if experiment == "toy":
        labels = toy_setup.get_labels()

    elif experiment == "imc":
        labels = imc_setup.get_labels()

    elif experiment == "citeseq":
        labels = citeseq_setup.get_labels()

    else:
        raise ValueError(f"Invalid experiment value: {experiment}")

    return labels
     
def main(data1, data2, experiment, callback, strategy, x_min, x_max, num_train_pts, optimise_iter, plot, ablate, ucb_scal):

    # Get labels based on experiment
    labels = get_labels(experiment)
            
    # Make acquired small dataset
    train_x = torch.rand(num_train_pts)
    train_x = torch.reshape(train_x, (train_x.shape[0], 1))
    log_vals = train_x*(x_max - x_min) + x_min
    train_x_full_range = log_vals

    # Compute y by calling correct true_f based on "experiment" 
    if experiment == "toy":
        train_y = toy_setup.true_f(train_x_full_range)
    elif experiment == "imc":
        train_y, ari, nmi = imc_setup.true_f(train_x_full_range, data1, data2)
    elif experiment == "citeseq":
        train_y, ari, nmi, hvgs = citeseq_setup.true_f(train_x_full_range, data1, data2)

    # Run a sequence of mobo experiments (with different strategies) based on "experiment"
    dicts_list = []
    success = True
    for strat in strategy:
        try:
            dicts_list.append(run_experiment(experiment, callback, strat, optimise_iter, train_x, train_y, data1, data2, toy_setup.true_f, x_min, x_max, labels, plot, ablate, ucb_scal))
        except RuntimeError as e:
            print(f"strategy {strat} failed with a RuntimeError\n")
            print(f"Error: {e}")
            success = False 
            pass
        except AssertionError as e:
            print(f"Assertion error\n")
            print(f"Error: {e}")
            successs = False
            pass

    mobo_output_dict = {k:v for d in dicts_list for k,v in d.items()} 
    return mobo_output_dict, success 

def main_crossval(data1_train, data2_train, data1_test, data2_test, fold, experiment, callback, strategy, x_min, x_max, num_train_pts, optimise_iter, plot, ablate, ucb_scal):

    # Get labels based on experiment
    labels = get_labels(experiment)
            
    # Make acquired small dataset
    train_x = torch.rand(num_train_pts)
    train_x = torch.reshape(train_x, (train_x.shape[0], 1))
    log_vals = train_x*(x_max - x_min) + x_min
    train_x_full_range = log_vals
    print(f"fold {fold}, {train_x}")

    # Compute y by calling correct true_f based on "experiment" 
    if experiment == "toy":
        train_y = toy_setup.true_f(train_x_full_range)
    elif experiment == "imc":
            train_y, ari, nmi = imc_setup.true_f(train_x_full_range, data1_train, data2_train)
    elif experiment == "citeseq":
        train_y, ari, nmi, hvgs = citeseq_setup.true_f(train_x_full_range, data1_train, data2_train)

    # Run a sequence of mobo experiments (with different strategies) based on "experiment"
    dicts_list = []
    success = True
    for strat in strategy:
        try:
            dicts_list.append(run_experiment_crossval(fold, experiment, callback, strat, optimise_iter, train_x, train_y, data1_train, data2_train, toy_setup.true_f, x_min, x_max, labels, data1_test, data2_test, plot, ablate, ucb_scal))
        except RuntimeError as e:
            print(f"strategy {strat} failed with a RuntimeError\n")
            print(f"Error: {e}")
            success = False 
            pass
        except AssertionError as e:
            print(f"Assertion error\n")
            print(f"Error: {e}")
            successs = False
            pass

    mobo_output_dict = {k:v for d in dicts_list for k,v in d.items()} 
    return mobo_output_dict, success 

def set_mobo_params(num_tasks, ucb_scal):
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks, 
                noise_constraint=gpytorch.constraints.GreaterThan(1e-2), has_global_noise=False)
    model_class = gp.MultitaskGPModel
    mll_class = gpytorch.mlls.ExactMarginalLogLikelihood
    if ucb_scal=="exhaustive":
        acq_fun = acq.mobo_ucb_scalarized
    elif ucb_scal=="mc":
        acq_fun = acq.mobo_ucb_scalarized_samples
    elif ucb_scal=="none":
        acq_fun = acq.mobo_acq
    else:
        print(f"{ucb_scal} value is invalid. Allowed: exhaustive, mc")
    acq_params = []
    acq_params.append(torch.tensor([1.]).repeat(num_tasks))
    acq_params.append(1.)
    return likelihood, model_class, mll_class, acq_fun, acq_params

def run_experiment(experiment, callback, strategy, optimise_iter, train_x, train_y, data1, data2, true_f, x_min, x_max, labels, plot, ablate, ucb_scal):
    num_tasks = train_y.shape[1]

    if experiment == "toy":
        true_f = toy_setup.true_f
    elif experiment == "imc":
        true_f = imc_setup.true_f
    elif experiment == "citeseq":
        true_f = citeseq_setup.true_f

    if strategy=='manatee' or strategy=='random prob':
        if strategy=='random prob' and ablate != "none":
            assert False, f"Ablating behaviours is only possible with manatee, not with {strategy}"
        likelihood, model_class, mll_class, acq_fun, acq_params = set_mobo_params(num_tasks, ucb_scal)
        print(f"Running {strategy}.\n")
        return bo.bayes_opt(
                    experiment, 
                    callback, 
                    optimise_iter, 
                    train_x,
                    train_y, 
                    data1,
                    data2, 
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
                    plot)

    elif strategy=='random loc':
        print(f"Running {strategy}.\n")
        return bo.bayes_opt_random(
                    experiment, 
                    callback, 
                    optimise_iter, 
                    train_x, 
                    train_y, 
                    data1, 
                    data2, 
                    true_f, 
                    x_min, 
                    x_max, 
                    labels, 
                    plot)

    elif strategy=='botorch':
        print(f"Running {strategy}.\n")
        return bo.bayes_opt_botorch(
                    experiment,
                    callback,
                    optimise_iter,
                    train_x,
                    train_y,
                    data1,
                    data2,
                    true_f,
                    x_min,
                    x_max,
                    labels,
                    plot) 

    elif strategy=='qparego':
        print(f"Running {strategy}.\n")
        return bo.bayes_opt_qparego(
                    experiment,
                    callback,
                    optimise_iter,
                    train_x,
                    train_y,
                    data1,
                    data2,
                    true_f,
                    x_min,
                    x_max,
                    labels,
                    plot) 

    elif strategy=='usemo':
        print(f"Running {strategy}.\n")
        return bo.bayes_opt_usemo(
                    experiment,
                    callback,
                    optimise_iter,
                    train_x,
                    train_y,
                    data1,
                    data2,
                    true_f,
                    x_min,
                    x_max)
    else: 
        print(f"Invalid strategy: {strategy}")

def run_experiment_crossval(fold, experiment, callback, strategy, optimise_iter, train_x, train_y, data1_train, data2_train, true_f, x_min, x_max, labels, data1_test, data2_test, plot, ablate, ucb_scal):
    num_tasks = train_y.shape[1]

    if experiment == "toy":
        true_f = toy_setup.true_f
    elif experiment == "imc":
        true_f = imc_setup.true_f
    elif experiment == "citeseq":
        true_f = citeseq_setup.true_f

    if strategy=='manatee' or strategy=='random prob':
        likelihood, model_class, mll_class, acq_fun, acq_params = set_mobo_params(num_tasks, ucb_scal)
        print(f"Running {strategy}.\n")
        return bo_cv.bayes_opt_crossval(
                    experiment, 
                    callback, 
                    optimise_iter, 
                    train_x,
                    train_y, 
                    data1_train,
                    data2_train, 
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
                    data1_test,
                    data2_test,
                    fold,
                    ablate,
                    plot)

    else: 
        print(f"Invalid strategy: {strategy}")

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run MOBO experiment.')
    parser.add_argument('--experiment', type=str, 
                        help='Type of experiment. Options: toy, imc, citeseq')
    parser.add_argument('--logging', type=str, default='nolog',
                        help='wandb / nolog')
    parser.add_argument('--strategy', nargs='+', default=['manatee'],
                        help='Strategy. Options: manatee, random prob, random loc, botorch, qparego, usemo')
    parser.add_argument('--x_min', type=float, 
                        help='Min range of x')
    parser.add_argument('--x_max', type=float, 
                            help='Max range of x')
    parser.add_argument('--num_train_pts', type=int, default=5,
                            help='Training set size')
    parser.add_argument('--optimise_iter', type=int, default=10,
                            help='Number of BO acquisition steps')
    parser.add_argument('--plot', type=bool, default=False,
                            help='Plot or not')
    parser.add_argument('--seed', default=10,
                            help='Set seed')
    parser.add_argument('--desc', default="default",
                            help='Experiment description')
    parser.add_argument('--crossval', default=False,
                            help='Run cross-val?')
    parser.add_argument('--ablate', type=str, default='none',
                            help='Which behaviour to ablate? Options: cor, noise, max')
    parser.add_argument('--ucb_scal', type=str, default="none",
                            help='Compute the AS acquisition function E[UCB(scal)] with MC sampling (mc) or exhaustively (exhaustive)?')
    args = parser.parse_args()
    log_settings(**vars(args))
