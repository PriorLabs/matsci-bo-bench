#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Train the pool head by policy gradient on simulated campaigns of synthetic pools.

Each iteration draws one pool and runs ``EPISODES`` campaigns on it in
lockstep from different initial designs, sampling picks from the head's
softmax. A pick's reward is what it adds to the benchmark's score: the AUC of
the metric over the 5-50% budgets. The baseline at each step is the mean
return-to-go of the other campaigns on the same pool.

    BACKBONE=v3.5 ITERS=300 OUT=results/rl python experiments/tabpfn35_acq/rl.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lockstep import _regressor, stop_length  # noqa: E402
from pool_head import PoolHead, tokens  # noqa: E402
from rules import Round  # noqa: E402
from synthetic import pool_mixed  # noqa: E402

from matbo.analysis.field import DENSE_BUDGETS, budget_columns  # noqa: E402

EPISODES = int(os.environ.get("EPISODES", "8"))
ARCH = {"width": 64, "depth": 2, "heads": 4}


def budget_weights(n: int, length: int) -> np.ndarray:
    """Weight of a gain made at evaluation t (0-based) in the 5-50% AUC."""
    cols = budget_columns(n, DENSE_BUDGETS)
    w = np.trapezoid(np.eye(len(cols)), DENSE_BUDGETS, axis=0) / np.ptp(DENSE_BUDGETS)
    out = np.zeros(length)
    for c, wc in zip(cols, w, strict=True):
        out[: c + 1] += wc
    return out


def episode_batch(reg, net, x, y, metric, rng):
    """Run EPISODES lockstep campaigns; return per-step (tokens, base, actions) and rewards."""
    n = len(y)
    length = stop_length(n, 0.5)
    k = int(np.ceil(0.05 * n))
    top = np.zeros(n, dtype=bool)
    top[np.argsort(-y)[:k]] = True
    span = float(y.max() - y.min()) or 1.0
    traces = [list(rng.choice(n, 2, replace=False)) for _ in range(EPISODES)]
    steps, rewards = [], []
    w = budget_weights(n, length)
    while len(traces[0]) < length:
        t = len(traces[0])
        masks = [np.isin(np.arange(n), tr, invert=True) for tr in traces]
        outs = reg.predict_batched(
            [x[tr] for tr in traces],
            [y[tr] for tr in traces],
            [x[m] for m in masks],
            output_type="full",
        )
        toks, bases, acts, rew = [], [], [], []
        for e, (tr, m, out) in enumerate(zip(traces, masks, outs, strict=True)):
            rest = np.flatnonzero(m)
            if out["logits"].shape[-1] == 1:
                tok, base = None, None
                a = int(rng.integers(len(rest)))
            else:
                r = Round(
                    out=out,
                    y_obs=y[tr],
                    n_pool=n,
                    top_frac=0.05,
                    rng=rng,
                    budget_frac=t / n,
                    metric=metric,
                )
                tok, base = tokens(r)
                with torch.no_grad():
                    logit = net(
                        torch.as_tensor(tok, dtype=torch.float32)[None],
                        torch.as_tensor(base, dtype=torch.float32)[None],
                    )[0]
                a = int(torch.distributions.Categorical(logits=logit).sample())
            pick = rest[a]
            if metric == "topcount":
                gain = float(top[pick])
            else:
                gain = max(0.0, float(y[pick] - y[tr].max())) / span
            toks.append(tok)
            bases.append(base)
            acts.append(a)
            rew.append(gain * w[t])
            tr.append(int(pick))
        steps.append((toks, bases, acts))
        rewards.append(rew)
    return steps, np.array(rewards)  # rewards [step, episode]


def main() -> int:
    out_dir = Path(os.environ["OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    iters = int(os.environ.get("ITERS", "300"))
    lo, hi = int(os.environ.get("MIN_N", "60")), int(os.environ.get("MAX_N", "300"))
    reg = _regressor(
        "cuda" if torch.cuda.is_available() else "cpu", 8, os.environ.get("BACKBONE", "v3.5")
    )
    torch.manual_seed(0)
    n_in = None
    net = opt = None
    ckpt = out_dir / "pool_head.pt"
    log_path = out_dir / "log.jsonl"
    start = 0
    if ckpt.exists():
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        n_in, start = blob["n_in"], blob["iter"]
        net = PoolHead(n_in, **ARCH)
        net.load_state_dict(blob["state"])
        opt = torch.optim.Adam(net.parameters(), lr=3e-4)
        opt.load_state_dict(blob["opt"])
    for it in range(start, iters):
        t0 = time.time()
        rng = np.random.default_rng([13, it])
        x, y = pool_mixed(rng, lo, hi)
        metric = "topcount" if it % 2 == 0 else "best-value"
        if net is None:
            out = reg.predict_batched([x[:5]], [y[:5]], [x[5:]], output_type="full")[0]
            r = Round(out=out, y_obs=y[:5], n_pool=len(y), top_frac=0.05, rng=rng, metric=metric)
            n_in = tokens(r)[0].shape[1]
            net = PoolHead(n_in, **ARCH)
            opt = torch.optim.Adam(net.parameters(), lr=3e-4)
        if it == start and net.sd.eq(1).all():
            # Token normalization from the first pool's rollout, which the zero readout
            # makes a rollout of the base rule (sampled).
            steps, rewards = episode_batch(reg, net, x, y, metric, rng)
            allt = np.concatenate(
                [np.concatenate([tk for tk in s[0] if tk is not None]) for s in steps]
            )
            net.mu.copy_(torch.as_tensor(allt.mean(0), dtype=torch.float32))
            net.sd.copy_(torch.as_tensor(allt.std(0) + 1e-3, dtype=torch.float32))
        else:
            net.eval()
            steps, rewards = episode_batch(reg, net, x, y, metric, rng)
        ret = rewards[::-1].cumsum(0)[::-1]  # return-to-go [step, episode]
        adv = ret - ret.mean(1, keepdims=True)
        scale = adv.std() + 1e-8
        net.train()
        opt.zero_grad()
        loss_sum, count = 0.0, 0
        for (toks, bases, acts), a_t in zip(steps, adv / scale, strict=True):
            keep = [e for e, tk in enumerate(toks) if tk is not None]
            if not keep:
                continue
            tok = torch.as_tensor(np.stack([toks[e] for e in keep]), dtype=torch.float32)
            base = torch.as_tensor(np.stack([bases[e] for e in keep]), dtype=torch.float32)
            logp = torch.log_softmax(net(tok, base), dim=-1)
            chosen = logp[torch.arange(len(keep)), torch.as_tensor([acts[e] for e in keep])]
            loss = -(torch.as_tensor(a_t[keep], dtype=torch.float32) * chosen).sum()
            loss.backward()
            loss_sum += loss.item()
            count += len(keep)
        for p in net.parameters():
            if p.grad is not None:
                p.grad /= max(count, 1)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        rec = {
            "iter": it,
            "n": len(y),
            "metric": metric,
            "return": float(rewards.sum(0).mean()),
            "loss": loss_sum / max(count, 1),
            "sec": time.time() - t0,
        }
        with log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        if it % 10 == 9 or it == iters - 1:
            torch.save(
                {
                    "n_in": n_in,
                    "arch": ARCH,
                    "state": net.state_dict(),
                    "opt": opt.state_dict(),
                    "iter": it + 1,
                },
                ckpt,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
