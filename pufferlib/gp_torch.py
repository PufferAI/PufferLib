"""
Drop-in replacement for GPyTorch exact GP training, CUDA implementation only.
"""

import numpy as np
import torch
import torch.nn as nn

from pufferlib import _C

class _MLLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, backend):
        ctx.backend = backend
        return params.new_tensor(backend.log_marginal_likelihood)

    @staticmethod
    def backward(ctx, grad_output):
        grads = torch.tensor(ctx.backend.mll_grad(),
                             dtype=grad_output.dtype, device=grad_output.device)
        return grad_output * grads, None


def _np32(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(x, dtype=np.float32)


class GaussianProcess(nn.Module):
    def __init__(self, dim, capacity,
                 lengthscale=1.0, outputscale=1.0, noise=1e-2, offset=1.0,
                 use_cuda=True):
        super().__init__()
        self._backend = _C.GaussianProcess(dim=dim, capacity=capacity,
                                lengthscale=lengthscale, outputscale=outputscale,
                                noise=noise, offset=offset)
        self.raw_lengthscale = nn.Parameter(
            torch.tensor(np.asarray(self._backend.raw_lengthscale), dtype=torch.float64))
        self.raw_outputscale = nn.Parameter(
            torch.tensor(self._backend.raw_outputscale, dtype=torch.float64))
        self.raw_noise = nn.Parameter(
            torch.tensor(self._backend.raw_noise, dtype=torch.float64))
        self.raw_offset = nn.Parameter(
            torch.tensor(self._backend.raw_offset, dtype=torch.float64))

    @property
    def lengthscale(self):
        return np.asarray(self._backend.lengthscale)

    @property
    def outputscale(self): return self._backend.outputscale

    @property
    def noise(self): return self._backend.noise

    @property
    def offset(self): return self._backend.offset

    @property
    def log_marginal_likelihood(self): return self._backend.log_marginal_likelihood

    @property
    def lengthscale_range(self):
        ells = self.lengthscale
        return float(np.min(ells)), float(np.max(ells))

    @property
    def n(self):        return self._backend.n

    @property
    def dim(self):      return self._backend.dim

    @property
    def capacity(self): return self._backend.capacity

    def fit(self, X, y):
        self._sync()
        self._backend.fit(_np32(X), _np32(y))

    def recompute(self):
        self._sync()
        self._backend.recompute()

    def mll(self, recompute=True):
        self._sync()
        if recompute:
            self._backend.recompute()
        params = torch.cat([self.raw_lengthscale,
                            self.raw_outputscale.unsqueeze(0),
                            self.raw_noise.unsqueeze(0),
                            self.raw_offset.unsqueeze(0)])
        return _MLLFunction.apply(params, self._backend)

    def predict(self, Xs):
        self._sync()
        means, vars_ = self._backend.predict(_np32(Xs))
        return torch.from_numpy(means), torch.from_numpy(vars_)

    def eval(self):
        result = super().eval()
        if hasattr(self, '_backend'):
            self._sync()
            self._backend.recompute()
        return result

    def save(self, path):
        self._sync()
        self._backend.save(path)

    @classmethod
    def load(cls, path, extra_cap=0, use_cuda=True):
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj._backend = _C.GaussianProcess.load(path, extra_cap)
        obj.raw_lengthscale = nn.Parameter(
            torch.tensor(np.asarray(obj._backend.raw_lengthscale), dtype=torch.float64))
        obj.raw_outputscale = nn.Parameter(
            torch.tensor(obj._backend.raw_outputscale, dtype=torch.float64))
        obj.raw_noise = nn.Parameter(
            torch.tensor(obj._backend.raw_noise, dtype=torch.float64))
        obj.raw_offset = nn.Parameter(
            torch.tensor(obj._backend.raw_offset, dtype=torch.float64))
        return obj

    def __repr__(self):
        ells = self.lengthscale
        if len(ells) == 1:
            ell_s = f"{ells[0]:.3g}"
        else:
            ell_s = f"[{np.min(ells):.3g}..{np.max(ells):.3g}]"
        return (f"<GaussianProcess dim={self.dim} n={self.n} cap={self.capacity} "
                f"ell={ell_s} sf={self.outputscale:.3g} "
                f"noise={self.noise:.3g}>")

    def _sync(self):
        self._backend.raw_lengthscale = self.raw_lengthscale.detach().cpu().numpy().astype(np.float32)
        self._backend.raw_outputscale = float(self.raw_outputscale.item())
        self._backend.raw_noise       = float(self.raw_noise.item())
        self._backend.raw_offset      = float(self.raw_offset.item())
