"""Simple Muon optimizer numerically matched to Lukas's HeavyBall implementation."""

import math
from collections.abc import MutableMapping
from typing import Optional

import torch
from torch import Tensor

from torch.optim.optimizer import (
    _disable_dynamo_if_unsupported,
    _params_doc,
    _to_scalar,
    Optimizer,
    ParamsT,
)


__all__ = ["Muon"]

def zeropower_via_newtonschulz5(G, eps=1e-7):
    G = G.clone()
    x = G
    if G.size(-2) > G.size(-1):
        x = x.mT

    x = x / torch.clamp(G.norm(dim=(-2, -1)), min=eps)
 
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        s = x @ x.mT
        y = c * s
        y.diagonal(dim1=-2, dim2=-1).add_(b)
        y = y @ s
        y.diagonal(dim1=-2, dim2=-1).add_(a)
        x = y @ x

    if G.size(-2) > G.size(-1):
        x = x.mT

    return x.to(G.dtype)

class Muon(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.0025,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        eps: float = 1e-8,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if momentum < 0.0:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def _init_group(
        self,
        group: MutableMapping,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        muon_momentum_bufs: list[Tensor],
    ):
        for p in group["params"]:
            if p.grad is None:
                continue

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    p.grad, memory_format=torch.preserve_format
                )
            muon_momentum_bufs.append(state["momentum_buffer"])

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eps = group["eps"]

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            muon_momentum_bufs: list[Tensor] = []
            lr = _to_scalar(lr)

            for p in group["params"]:
                if p.grad is None:
                    continue

                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p.grad, memory_format=torch.preserve_format
                    )
                muon_momentum_bufs.append(state["momentum_buffer"])

            for i, param in enumerate(params_with_grad):

                grad = grads[i]

                buf = muon_momentum_bufs[i]
                buf.mul_(momentum)
                buf.add_(grad)
                grad.add_(buf*momentum)

                if grad.ndim >= 2:
                    grad = grad.view(grad.shape[0], -1)
                    grad = zeropower_via_newtonschulz5(grad) # original has hardcoded steps and eps
                    grad *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5 # Matches heavyball and Keller

                param.mul_(1 - lr * weight_decay)
                param.sub_(lr*grad.view(param.shape))

        return loss
