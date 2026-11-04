#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Gaussian-process surrogates: BoTorch ``SingleTaskGP`` over seven kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.models.transforms.input import ChainedInputTransform, Warp
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from matbo.methods.base import Posterior, ScoreRequest

#: kernel -> [failed fits, attempted fits], reported at the end of a run.
FIT_FAILURES: dict[str, list[int]] = {}
#: Above this share of failed fits a kernel is flagged in the report.
FAILURE_ALARM = 0.10


def fit_failure_report(clear: bool = True) -> list[str]:
    """Lines describing this run's fit failures, worst first. Empty if none."""
    lines = []
    for name, (fails, tries) in sorted(FIT_FAILURES.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
        if not fails:
            continue
        rate = fails / tries
        note = f"  UNUSABLE: its curve is not a fitted GP-{name}" if rate > FAILURE_ALARM else ""
        lines.append(f"GP-{name}: {fails}/{tries} fits fell back to the prior ({100 * rate:.0f}%){note}")
    if clear:
        FIT_FAILURES.clear()
    return lines


class MLPKernel(Kernel):
    """Arccosine (MLP) kernel, as GPy's ``kern.MLP``.

    k(x, x') = (2/pi) asin( (x^T W x' + b) / sqrt((x^T W x + b + 1)(x'^T W x' + b + 1)) ),
    with W = diag(weight_variance), ARD when it has one entry per dimension.
    """

    is_stationary = False

    def __init__(self, ard_num_dims=None, **kwargs):
        super().__init__(ard_num_dims=ard_num_dims, **kwargs)
        shape = (*self.batch_shape, 1, ard_num_dims or 1)
        self.register_parameter("raw_weight_variance", torch.nn.Parameter(torch.zeros(*shape)))
        self.register_constraint("raw_weight_variance", Positive())
        self.register_parameter("raw_bias_variance", torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
        self.register_constraint("raw_bias_variance", Positive())

    @property
    def weight_variance(self):
        return self.raw_weight_variance_constraint.transform(self.raw_weight_variance)

    @property
    def bias_variance(self):
        return self.raw_bias_variance_constraint.transform(self.raw_bias_variance)

    def forward(self, x1, x2, diag=False, **_params):
        w, b = self.weight_variance, self.bias_variance
        d1 = (x1 * x1 * w).sum(-1, keepdim=True) + b + 1.0
        d2 = (x2 * x2 * w).sum(-1, keepdim=True) + b + 1.0
        if diag:
            cross = (x1 * x2 * w).sum(-1, keepdim=True) + b
            ratio = cross / torch.sqrt(d1 * d2)
            return torch.asin(ratio.clamp(-1 + 1e-7, 1 - 1e-7)).squeeze(-1) * (2 / math.pi)
        cross = (x1 * w) @ x2.transpose(-1, -2) + b
        ratio = cross / torch.sqrt(d1 * d2.transpose(-1, -2))
        return torch.asin(ratio.clamp(-1 + 1e-7, 1 - 1e-7)) * (2 / math.pi)


KERNELS = {
    "ARD": lambda d: MaternKernel(nu=2.5, ard_num_dims=d),
    "M32": lambda d: MaternKernel(nu=1.5, ard_num_dims=d),
    "M12": lambda d: MaternKernel(nu=0.5, ard_num_dims=d),
    "RBF": lambda d: RBFKernel(ard_num_dims=d),
    "iso": lambda _d: MaternKernel(nu=2.5),
    "MLP": lambda d: MLPKernel(ard_num_dims=d),
}


def _fit_or_fall_back(gp: SingleTaskGP, kernel: str) -> None:
    """Fit once with a noise floor and capped iterations; on failure keep the prior.

    Greedy selection clusters candidates, so some fits go singular. Dropping
    those seeds would score the GP only where it worked; keeping the unfitted
    prior keeps every seed, and the failure rate is reported per kernel.
    """
    tally = FIT_FAILURES.setdefault(kernel, [0, 0])
    tally[1] += 1
    gp.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-4))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    try:
        fit_gpytorch_mll(mll, max_attempts=1, optimizer_kwargs={"options": {"maxiter": 150}})
    except Exception as exc:  # noqa: BLE001 - any fit failure falls back to the prior
        tally[0] += 1
        if tally[0] == 1:
            print(f"    WARNING GP-{kernel}: fit failed, falling back to prior hyperparameters -- "
                  f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}", flush=True)


@dataclass
class GPSurrogate:
    """``SingleTaskGP`` with inputs normalized to the unit cube and y standardized.

    ``warp`` adds BoTorch's Kumaraswamy input warping after the normalization.
    """

    kernel: str = "ARD"
    warp: bool = False
    name: str = field(default="", init=False)

    def __post_init__(self):
        if self.kernel not in KERNELS:
            raise ValueError(f"unknown kernel {self.kernel!r}; have {sorted(KERNELS)}")
        self.name = f"GP-{self.kernel}" + ("-warp" if self.warp else "")

    def fit(self, req: ScoreRequest) -> Posterior:
        tx = torch.as_tensor(req.x_obs, dtype=torch.double)
        ty = torch.as_tensor(req.y_obs, dtype=torch.double).unsqueeze(-1)
        d = tx.shape[1]
        transform = (
            ChainedInputTransform(normalize=Normalize(d=d), warp=Warp(d=d, indices=list(range(d))))
            if self.warp
            else Normalize(d=d)
        )
        gp = SingleTaskGP(
            tx, ty,
            covar_module=ScaleKernel(KERNELS[self.kernel](d)),
            input_transform=transform,
            outcome_transform=Standardize(m=1),
        )
        _fit_or_fall_back(gp, self.name.removeprefix("GP-"))

        cand = torch.as_tensor(req.x_cand, dtype=torch.double)
        with torch.no_grad():
            post = gp.posterior(cand)
            mean = post.mean.squeeze(-1).numpy()
            sd = post.variance.squeeze(-1).sqrt().numpy()

        def exact_ei(best_f: float) -> np.ndarray:
            # LogEI is monotone in EI, so the pool argmax is the same, and it does
            # not underflow to 0 far from the incumbent.
            acqf = LogExpectedImprovement(gp, best_f=best_f)
            with torch.no_grad():
                return acqf(cand.unsqueeze(1)).numpy()

        return Posterior(mean=mean, sd=sd, exact_ei=exact_ei)
