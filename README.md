# GP-Ouroboros: Using Gaussian Processes to model the log-likelihood in Bayesian Hyperparameter Inference for Gaussian Processes
This project explores the use of Gaussian Processes (GPs) to model the
log-marginal likelihood in fully Bayesian inference on Gaussian Processes. Our
focus for now is on the actual sampling process itself; perhaps in the future
we will move on to active learning for smart likelihood evaluations. The motivation
is that for large GPs the log-marginal likelihood and its gradients can be
expensive to compute. Thus we may turn to surrogate models. One such possibility
is to use GPs themselves as the surrogate model. We consider three possible
algorithms for sampling where the log-likelihood is modelled by a GP
- Using the GP mean as our surrogate model for NUTS (Stan)
- Using an adjusted version to correct for the fact that the likelihood is log-normal
- Sampling gradients instead of using function evaluations in uncorrected
Langevin dynamics

Additionally, where computationally feasible we run NUTS using evaluations of the
true log likelihood.

For the base model we consider a simple a radial basis function (RBF) GP with 
Automatic Relevance Determination (ARD) and learnable output scale
but fixed noise. Length-scales and output scales are given a standard log-normal prior and
inference is done on the log-scale. For the surrogate model we also use an RBF
GP with ARD on the log-parameter space. Hyperparameters for the surrogate model
are chosen with type II maximum likelihood estimation (MLE Type II).

For each experiment we control three parameters:
- `n`: The number of observations in the base model. This directly controls the
both the difficulty of evaluating the log marginal likelihood, and the balance
between prior and likelihood in hyperparameter inference
- `d`: The dimensionality of the problem. In addition to making the inference
problem more complicated by introducing more parameters to infer, this also moves
likelihood evaluations further apart (the curse of dimensionality strikes!)
- `m`: The number of likelihood evaluations used to construct the surrogate model.

The surrogate model log parameter space is restricted to `[-5, 5]` in each dimension
(i.e. a hypercube) to prevent exploding or disappearing real paramters. For every pair of $n$ and $d$ the underlying model is the same for each $m$.

This repository contains one directory for each set of results. The directory
format should be read as `n-d-m-seed`. Summaries of the pair plots and trace
plots can be found in `trace_plots.pdf` and `pair_plots.pdf`. Summary statistics
such as ESS and Rhat can be found in each folder's `summary.csv`. A more
detailed discussion of the background and theory can be found in `report.pdf`.

## AI Disclosure
Chat GPT was used to generate code to compile plots into pdfs. All other code is
human written.