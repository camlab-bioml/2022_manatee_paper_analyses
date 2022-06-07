# Multi-objective Bayesian optimization with heuristic objectives
This repository contains the code implementing MANATEE used for the analyses in the associated paper. 

MANATEE (Multi-objective bAyesiaN optimizAtion wiTh hEuristic objEctives) is a multi-objective Bayesian optimization method, which automatically up- or downweights heuristic objectives based on the properties of their posterior functional form. These properties are specified as desirable or non-desirable *behaviours*, which reflect user's expectations of what a useful heuristic objective should look like. We propose for MANATEE to be used for parameter optimization in biomedical and molecular data analysis pipelines, where objectives being optimized correspond to heuristic measures of a pipeline's success. We used MANATEE to optimize the cofactor normalization parameter for the analysis of imaging mass cytometry (IMC) data and the proportion of highly variable genes for the analysis of single-cell RNA-sequencing (scRNA-seq) data.

## Dependencies

- Python 3.8
- scipy
- matplotlib
- scikit-learn
- PyTorch 1.9.0
- gpytorch
- botorch
- scanpy
- leidenalg
- wandb

## Quickstart 

MANATEE is run by executing the script `mobo_experiment.py` with corresponding arguments. The code supports three experiments discussed in the paper (toy, IMC, scRNA-seq) by implementing the corresponding pipelines required for new acquisitions. MANATEE can be executed by specifying the experiment and the parameter optimization bounds, for example:

```
python mobo_experiment.py --experiment imc --x_min 1 --x_max 100
```

There are additional optional arguments described in the help message. By default, `mobo_experiment.py` will execute only MANATEE (specifically, the MANATEE-SA version), but the code also supports other methods considered in the paper (MANATEE-AS, RS and RA baselines, qNEHVI with approximate hypervolume computation, qNParEGO). These can be executed by specifying the `--strategy` argument. MANATEE-AS can be executed with the default `--strategy` and `--ucb_scal` set to `True`.

The code supports [Weights and Biases](https://wandb.ai) integration (with `--logging` set to `wandb`) which tracks acquisitions, meta-objectives (ARI, NMI), and objective inclusion probabilities and behaviours. These can be also accessed from the dictionary returned by the function `main`, along with the acquisition function values at each step and the final acquired datasets.

## Authors

Alina Selega, Kieran R. Campbell

Lunenfeld-Tanenbaum Research Institute, Vector Institute, University of Toronto
