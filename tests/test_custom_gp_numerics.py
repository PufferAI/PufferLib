"""
Custom CUDA GP vs GPyTorch

Hyperparameters are fixed identically, just comparing the raw numerical results and timings for training and prediction.
"""

import time, sys, math
import numpy as np

def make_data(n, dim=1, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-5, 5, size=(n, dim))
    y = np.sin(X[:, 0]) + 0.1 * rng.randn(n)
    return X.astype(np.float64), y.astype(np.float64)

def make_test(m=200, dim=1):
    xs = np.linspace(-6, 6, m).reshape(-1, 1)
    if dim > 1:
        xs = np.hstack([xs, np.zeros((m, dim - 1))])
    return xs.astype(np.float64)

def bench_custom(X, y, Xs, hp):
    from pufferlib.gp_torch import GaussianProcess
    import torch  # ensure CUDA context is warm before timing

    # Warm-up: a tiny fit so the first CUDA call overhead isn't charged
    # to the benchmark (driver init, JIT, etc.)
    _gp = GaussianProcess(dim=X.shape[1], capacity=1,
                          lengthscale=hp["lengthscale"],
                          outputscale=hp["outputscale"],
                          noise=hp["noise"])

    t0 = time.perf_counter()
    gp = GaussianProcess(dim=X.shape[1], capacity=len(X),
                          lengthscale=hp["lengthscale"],
                          outputscale=hp["outputscale"],
                          noise=hp["noise"])
    
    print(gp)
    gp.fit(X, y)
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    means, vars_ = gp.predict(Xs)
    t_pred = time.perf_counter() - t0

    return means.numpy(), vars_.numpy(), gp.log_marginal_likelihood, t_train, t_pred

def bench_gpytorch(X, y, Xs, hp):
    import torch, gpytorch

    class ExactGP(gpytorch.models.ExactGP):
        def __init__(self, tx, ty, lik):
            super().__init__(tx, ty, lik)
            from gpytorch.kernels import (MaternKernel, PolynomialKernel,
                                          AdditiveKernel, ScaleKernel)
            self.mean_module  = gpytorch.means.ZeroMean()
            matern = MaternKernel(nu=1.5, ard_num_dims=tx.shape[-1])
            linear = PolynomialKernel(power=1)
            self.covar_module = ScaleKernel(AdditiveKernel(linear, matern))
        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x))

    Xt  = torch.from_numpy(X).double()
    yt  = torch.from_numpy(y).double()
    Xst = torch.from_numpy(Xs).double()

    t0 = time.perf_counter()
    lik   = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGP(Xt, yt, lik)
    model.double()
    model.covar_module.base_kernel.kernels[1].lengthscale  = hp["lengthscale"]
    model.covar_module.outputscale                         = hp["outputscale"]
    lik.noise                                              = hp["noise"]
    model.covar_module.base_kernel.kernels[0].raw_offset.data.fill_(
        math.log(math.expm1(hp["offset"])))
    t_train = time.perf_counter() - t0

    # Force exact Cholesky (default CG cutoff is 800)
    chol_size = max(len(yt) + 1, 801)
    model.eval(); lik.eval()
    t0 = time.perf_counter()
    with torch.no_grad(), \
         gpytorch.settings.fast_pred_var(), \
         gpytorch.settings.max_cholesky_size(chol_size):
        latent = model(Xst)
        means  = latent.mean.numpy()
        vars_  = latent.variance.numpy()
    t_pred = time.perf_counter() - t0

    model.train(); lik.train()
    with torch.no_grad(), gpytorch.settings.max_cholesky_size(chol_size):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
        mll = mll(model(Xt), yt).item()

    return means, vars_, mll, t_train, t_pred

def run(n, dim=1):
    hp   = dict(lengthscale=1.0, outputscale=1.0, noise=0.01, offset=math.log(2))
    X, y = make_data(n, dim)
    Xs   = make_test(200, dim)

    rows = {}

    try:
        rows["CUDA"] = bench_custom(X, y, Xs, hp)
    except ImportError:
        pass

    rows["GPyTorch"] = bench_gpytorch(X, y, Xs, hp)

    ref_means = rows["GPyTorch"][0]
    ref_vars  = rows["GPyTorch"][1]

    print(f"\n{'='*84}")
    print(f"  n={n}  dim={dim}")
    print(f"{'='*84}")
    print(f"  {'':22s} {'train':>8s} {'predict':>8s} {'total':>8s}"
          f" {'MLL':>12s} {'|err_mean|':>10s} {'|err_var|':>10s}")

    for label, (m_, v_, mll, tt, tp) in rows.items():
        me = np.max(np.abs(m_ - ref_means))
        ve = np.max(np.abs(v_ - ref_vars))
        print(f"  {label:22s} {tt:7.4f}s {tp:7.4f}s {tt+tp:7.4f}s"
              f" {mll:12.4f} {me:9.2e} {ve:9.2e}")

    print(f"  {'':22s} (GPyTorch: exact Cholesky forced for n>800, errors vs GPyTorch)")

if __name__ == "__main__":
    sizes = [100, 200, 1000, 2000]
    dims  = [1, 5]
    if len(sys.argv) > 1:
        sizes = [int(s) for s in sys.argv[1:]]
    for n in sizes:
        for d in dims:
            run(n, dim=d)
