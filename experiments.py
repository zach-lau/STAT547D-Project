import torch
import random
import math
import matplotlib.pyplot as plt
import torch.optim as optim
import arviz as az
import os
from datetime import datetime
import cmdstanpy as csp
torch.set_default_dtype(torch.float64)

def printtime():
    print(datetime.now().time())

def rbf(x, l, sf2, sn2):
    squares = torch.sum(((x.unsqueeze(1)-x.unsqueeze(0))/l)**2, dim=2)
    K = sf2*torch.exp(-1/2*squares) + sn2*torch.eye(len(x)) # Stability
    return K

def generate_gp(l, sf2, sn2, x):
    """ Generate sample sample from an rbf gp with given lengthscales
    l1: lengthscalles
    sf2: output scale
    x: matrix of points to sample at
    
    Assume the covariance matrix is not too close to statinary i.e. relatively
    large lengthscales. If it's closer to stationary RFF might be better
    """
    K = rbf(x, l, sf2, sn2)
    L = torch.linalg.cholesky(K)
    y = torch.sqrt(sf2)*(L @ torch.randn(len(x)))
    return y

def gp_lml(x, y, l, sf2, sn2):
    """ Calculate log-marginal-likelihood given specific parameters """
    K = rbf(x, l, sf2, sn2)
    comp = y @ torch.linalg.solve(K, y)
    fit = torch.logdet(K)
    lml = -1/2*(comp+fit)
    return lml

def cross_mean(zeta, alpha, l, sf2):
    """Compute the cross covariance of zeta and grad zeta with alphas"""
    z = zeta.unsqueeze(0).unsqueeze(1)
    a = alpha.unsqueeze(0)
    squares = torch.sum(((z-a)/l)**2, dim=2)
    K1 = sf2*torch.exp(-1/2*squares) # 1 x m matrix
    return K1

def cross_rbf(zeta, alpha, l, sf2):
    """Compute the cross covariance of zeta and grad zeta with alphas"""
    z = zeta.unsqueeze(0).unsqueeze(1)
    a = alpha.unsqueeze(0)
    squares = torch.sum(((z-a)/l)**2, dim=2)
    K1 = sf2*torch.exp(-1/2*squares) # 1 x m matrix
    K2 = K1*((zeta-alpha)/l**2).T 
    return torch.cat((K1, K2), dim = 0)

def grad_rbf(l, sf2):
    """ Compute the prior covariance of the observation values and its
    gradients """
    return sf2*torch.diag(torch.tensor([1+1e-4/sf2, *(1/l**2)]))

def sample_one(zeta, Kinv, lp, sp2, alpha, gamma):
    k = cross_rbf(zeta, alpha, lp, sp2)
    kKi = k @ Kinv
    mean =  kKi @ gamma
    kss = grad_rbf(lp, sp2)
    variance = kss - kKi @ k.T
    L = torch.linalg.cholesky(variance)
    out = mean + L @ torch.randn(len(mean)) # GP prediction
    return out

def run(n, d, m, seed, warmup = 1000, iterations = 1000, iterations_true = None):
    #########
    # Setup #
    #########
    if iterations_true is None:
        iterations_true = iterations
    print(f"Setting up with {n = }, {d = }")
    name = f"{n}-{d}-{m}-{seed}"
    os.makedirs(name, exist_ok=True)
    names = ["langevin", "mean", "adjusted", "gold"]
    for r in names:
        os.makedirs(os.path.join(name, r), exist_ok=True)
    torch.manual_seed(seed)
    noise = 1e-2
    x = torch.rand(n, d)*2-1 # Scale from -1 to 1
    theta = torch.randn(d+1) # Underlying parameter
    l = torch.exp(theta[:d])
    sf2= torch.exp(theta[-1])
    bounds = 5 # bounds on theta

    #########
    # Prior #
    #########
    def logptheta(theta):
        return -(d+1)/2*math.log(2*torch.pi) - 1/2*torch.sum(theta*theta)
    def grad_logptheta(theta):
        return -theta
    y = generate_gp(l, sf2, noise, x)

    ##########################
    # Likelihood evaluations #
    ##########################
    # Generate random evaluations of the likelihood to build our GP model
    # alpha = torch.randn(m, d+1) # Generate theta from prior
    alpha = torch.rand(m, d+1)*2*bounds - bounds
    beta = torch.tensor([gp_lml(x,y,torch.exp(a[:d]),torch.exp(a[-1]), noise) for a in alpha])

    if d == 1:
        # Can't really plot in higher dimensions
        plt.scatter(theta[0], theta[1], c="r")
        plt.scatter(alpha[:,0], alpha[:,1], c=beta)
        plt.colorbar()
        plt.savefig(os.path.join(name, "train_points.png"))
        plt.close()
    ##########################
    # Train likelihood model #
    ##########################

    # Optimize over theta
    # Scale and center likelihood observations
    mu = torch.mean(beta)
    s2 = torch.var(beta)
    gamma = (beta - mu)/math.sqrt(s2)

    # Metaparamters
    # Extra paramter since we model over output scale too
    xi = torch.randn(d+2, requires_grad=True)
    optimizer = optim.Adam((xi,), lr=1e-2)
    losses = []
    this_noise = 1e-2

    # TODO could add multistart, might get us a better model
    for _ in range(1000):
        this_l = torch.exp(xi[:d+1])*math.sqrt(d) # Scale for dimension
        this_s2 = torch.exp(xi[-1])
        loss = -gp_lml(alpha, gamma, this_l, this_s2, this_noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(f"Iteration {i}. Loss = {loss.item()}")
        losses.append(loss.item())
    plt.plot(losses)
    plt.savefig(os.path.join(name,"loss.png"))
    plt.close()

    xi.detach_()
    lp = torch.exp(xi[:d+1])*math.sqrt(d) # Scale for dimension
    sp2 = torch.exp(xi[-1])
    K = rbf(alpha, lp, sp2, this_noise)
    Kinv = torch.linalg.inv(K) # Precompute inverse

    ###############################
    # Visualize mean and variance #
    ###############################
    if d == 1:
        weights = Kinv @ gamma
        def gp_mean(x):
            k = cross_mean(x, alpha, lp, sp2)
            return k @ weights
        def gp_var(x):
            k = cross_mean(x, alpha, lp, sp2)
            return sp2 - (k @ (Kinv @ k.T)).item()
        
        grid_bounds = bounds
        grid = torch.linspace(-grid_bounds, grid_bounds, 20)
        x_test = torch.stack([a.flatten() for a in torch.meshgrid(grid, grid)], dim = 1)
        y_test = [gp_mean(x)*torch.sqrt(s2)+mu for x in x_test]
        plt.scatter(x_test[:,0], x_test[:,1], c=y_test)
        plt.colorbar()
        plt.scatter(alpha[:,0], alpha[:,1], color="r")
        plt.savefig(os.path.join(name, "mean.png"))
        plt.close()

        var_test = [torch.sqrt(gp_var(x)*s2) for x in x_test]
        plt.scatter(x_test[:,0], x_test[:,1], c=var_test)
        plt.colorbar()
        plt.scatter(alpha[:,0], alpha[:,1], color="r")
        plt.savefig(os.path.join(name, "sd.png"))
        plt.close()

        log_mean = [gp_mean(x)*torch.sqrt(s2)+mu + 1/2*gp_var(x)*s2 for x in x_test]
        plt.scatter(x_test[:,0], x_test[:,1], c=log_mean)
        plt.colorbar()
        plt.scatter(alpha[:,0], alpha[:,1], color="r")
        plt.savefig(os.path.join(name, "logmean.png"))
        plt.close()
        
        if n < 200:
            true_lml = [gp_lml(x, y, torch.exp(t[:d]), torch.exp(t[-1]), this_noise) for t in x_test]
            plt.scatter(x_test[:,0], x_test[:,1], c=true_lml)
            plt.colorbar()
            plt.scatter(alpha[:,0], alpha[:,1], color="r")
            plt.savefig(os.path.join(name, "truelml.png"))
            plt.close()

    ######################
    # Langevin Dynamics  #
    ######################
    samples = []

    print("Running Langevin Dyanmics...")
    printtime()
    chains = []
    for _ in range(4):
        zeta = torch.randn(d+1)
        tau = 0.1 /math.sqrt(s2) # Scale approximately by gradient size
        draws = []
        for _ in range(warmup + iterations):
            out = sample_one(zeta, Kinv, lp, sp2, alpha, gamma)
            grad_sample = out[1:]*math.sqrt(s2)
            grad_prior = grad_logptheta(zeta) # /math.sqrt(s2)
            grad = grad_prior + grad_sample
            cand = zeta + tau*grad + math.sqrt(2*tau)*torch.randn(d+1)
            if torch.all(cand > -bounds) and torch.all(cand < bounds):
                zeta = cand
            draws.append(zeta)
        chains.append(draws[warmup:])

    printtime()
    print("Done Langevin")
    sample_ld = az.from_dict(
        posterior = {
            "theta": chains
        }
    )
    samples.append(sample_ld)

    #############
    # Mean NUTS #
    #############
    print("Running NUTS with GP mean")
    data = {
        "m": m, # Number of data points in surrogate model
        "d": d, # Dimension of data points
        "Kinv": Kinv.numpy(), # Inverse covariance matrix (includig noise)
        "alpha": alpha.numpy(),
        "gamma": gamma.numpy(), # likelihood at observations
        "l": lp.numpy(), # Surrogate length scale vector length d+1
        "sf2": sp2.numpy(), # Surrogate out put scale
        "mu": mu.numpy(), # Centering for likelihood
        "s2": s2.numpy(), # Scaling factor
        "bound": bounds,
    }
    model  = csp.CmdStanModel(stan_file = "mean.stan")
    printtime()
    sample_mean = model.sample(data = data, seed = 123, chains = 4, iter_sampling = iterations,
                      iter_warmup=warmup, show_progress=False, show_console=False)
    printtime()
    print("Done mean")
    samples.append(sample_mean)

    #################
    # Adjusted mean #
    #################
    print("Running with adjusted mean")
    model  = csp.CmdStanModel(stan_file = "lognorm.stan")
    printtime()
    sample_adjusted = model.sample(data = data, seed = 123, chains = 4, iter_sampling = iterations,
                      iter_warmup=warmup, show_progress=False, show_console=False)
    printtime()
    print("Done adjusted")
    samples.append(sample_adjusted)

    ###############
    # HMC on true #
    ###############
    # We avoid running HMC on the true target for larger cases because it
    # takes too long
    if n < 200:
        print("Running HMC on true target")
        model  = csp.CmdStanModel(stan_file = "lml.stan")
        data = {
            "n":n,
            "d":d,
            "x":x.numpy(),
            "y":y.numpy(),
            "sn2":noise,
            "bound": bounds,
        }
        printtime()
        sample_gold = model.sample(data = data, seed = 123, chains = 4, iter_sampling = iterations_true,
                        iter_warmup=warmup, show_progress=False, show_console=False)
        printtime()
        print("Done true")
        samples.append(sample_gold)

    #############################
    # Visualize and diagnostics #
    #############################

    print("Saving figures and summaries")
    for i, c in enumerate(samples):
        savedir = os.path.join(name, names[i])
        az.plot_trace(c, var_names= ["theta"])
        plt.savefig(os.path.join(savedir, "trace.png"))
        plt.close()
        az.plot_pair(c, var_names = ["theta"])
        plt.savefig(os.path.join(savedir, "pairs.png"))
        plt.close()
        summary = az.summary(c, var_names=["theta"])
        summary.round(2).to_csv(os.path.join(savedir, "summary.csv"))

if __name__ == "__main__":
    random.seed(123)
    ns = [10, 100, 1000]
    ms = [10, 100]
    ds = [1, 5, 50]
    runs = []
    for n in ns:
        for d in ds:
            for m in ms:
                runs.append((n,d,m))
    random.shuffle(runs)
    print("Run order is")
    print(runs)
    for r in runs:
        run(*r, 123, 1000, 9000, 1000)
