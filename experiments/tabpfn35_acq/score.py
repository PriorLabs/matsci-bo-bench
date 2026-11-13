#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Score new arms against the paper's 56-arm field.

Each new arm joins the field on its own (56 + 1 arms), so new arms never move
each other's NI. NI and seed AUCs read budgets 5-50% only, so traces cut at
half the pool score exactly as full ones do.

    python experiments/tabpfn35_acq/score.py liang results/step0 [--seeds first|second|all]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from matbo.analysis.field import DENSE_BUDGETS, FIELD, budget_columns, normalize
from matbo.analysis.posthoc import seed_auc
from matbo.analysis.record import Record
from matbo.runner import curve_from_trace


def _scale(record: Record, pool) -> float:
    if record.protocol.metric == "topcount":
        return 100.0 / int(pool.top_mask(record.protocol.top_frac).sum())
    return 1.0


def new_arm_blocks(record: Record, folder: Path) -> dict[str, dict[str, np.ndarray]]:
    """arm -> campaign -> [seed, evaluation] curves, rows in protocol seed order."""
    out: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for path in sorted(folder.rglob("*.trace.csv")):
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            campaign = row["campaign"]
            pool = record.campaign(campaign)
            if row["pool_fingerprint"] != pool.fingerprint():
                raise ValueError(f"{path}: pool moved")
            keys = sorted((k for k in row if k[:1] == "s" and k[1:].isdigit()), key=lambda k: int(k[1:]))
            trace = np.array([int(row[k]) for k in keys])
            curve = _scale(record, pool) * curve_from_trace(trace, pool, record.protocol)
            out.setdefault(row["method"], {}).setdefault(campaign, {})[int(row["seed"])] = curve
    blocks = {}
    for arm, by_c in out.items():
        blocks[arm] = {}
        for campaign, by_s in by_c.items():
            seeds = [s for s in record.protocol.seeds if s in by_s]
            blocks[arm][campaign] = (seeds, np.vstack([by_s[s] for s in seeds]))
    return blocks


def field_blocks(record: Record) -> dict[str, np.ndarray]:
    """campaign -> [FIELD arm, seed, evaluation], cut at 50% of the pool (+1)."""
    out = {}
    for campaign in record.campaigns:
        n = record.campaign(campaign).n
        width = int(np.rint(0.5 * n)) + 1
        out[campaign] = np.stack([record.block(a, campaign)[:, :width] for a in FIELD])
    return out


def ni_auc(blocks: dict[str, np.ndarray], record: Record) -> np.ndarray:
    """[campaign, arm] NI averaged over the linear 5-50% budgets."""
    per = []
    for campaign, block in blocks.items():
        n = record.campaign(campaign).n
        cols = budget_columns(n, DENSE_BUDGETS)
        ni = normalize(block[:, :, cols].mean(axis=1).T)  # [budget, arm]
        per.append(np.trapezoid(ni, DENSE_BUDGETS, axis=0) / np.ptp(DENSE_BUDGETS))
    return np.array(per)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--seeds", default="all", choices=["all", "first", "second"])
    ap.add_argument("--vs", default="", help="comma-separated arms to test against (default: best)")
    args = ap.parse_args()

    record = Record.load(args.benchmark)
    seeds = list(record.protocol.seeds)
    half = len(seeds) // 2
    keep = {"all": seeds, "first": seeds[:half], "second": seeds[half:]}[args.seeds]
    fb = field_blocks(record)
    new = new_arm_blocks(record, args.folder)

    rows, tests = [], []
    for arm, by_c in sorted(new.items()):
        if set(by_c) != set(record.campaigns):
            print(f"skip {arm}: has {sorted(by_c)}")
            continue
        blocks = {}
        for campaign in record.campaigns:
            have, curves = by_c[campaign]
            sel = [i for i, s in enumerate(seeds) if s in keep and s in have]
            mine = [have.index(seeds[i]) for i in sel]
            if len(sel) != len(keep):
                break
            width = fb[campaign].shape[2]
            blocks[campaign] = np.concatenate(
                [fb[campaign][:, sel, :width], curves[None, mine, :width]], axis=0
            )
        else:
            names = [*FIELD, arm]
            per = ni_auc(blocks, record)  # [campaign, arm]
            overall = pd.Series(np.nanmean(per, axis=0), index=names).drop("Random")
            ranked = overall.sort_values(ascending=False)
            best_old = next(a for a in ranked.index if a != arm)
            vs = args.vs.split(",") if args.vs else [best_old]
            row = {
                "arm": arm,
                "NI": overall[arm],
                "rank": int(list(ranked.index).index(arm)) + 1,
                "best_existing": best_old,
                "best_existing_NI": overall[best_old],
            }
            for c, campaign in enumerate(record.campaigns):
                row[campaign] = per[c, names.index(arm)]
            rows.append(row)
            for ref in vs:
                j = names.index(ref)
                d_ni = per[:, names.index(arm)] - per[:, j]
                wins = 0
                for c, campaign in enumerate(record.campaigns):
                    n = record.campaign(campaign).n
                    a = seed_auc(_pad(blocks[campaign][-1], n), n)
                    b = seed_auc(_pad(blocks[campaign][j], n), n)
                    d = a - b
                    p = stats.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
                    wins += int(d.mean() > 0 and p < 0.05) - int(d.mean() < 0 and p < 0.05)
                    tests.append(
                        {"arm": arm, "vs": ref, "campaign": campaign, "mean_diff": d.mean(),
                         "wilcoxon_p": p, "dNI": d_ni[c]}
                    )
                p_c = stats.wilcoxon(d_ni).pvalue if np.any(d_ni != 0) else 1.0
                tests.append(
                    {"arm": arm, "vs": ref, "campaign": "ALL (campaign NI)", "mean_diff": d_ni.mean(),
                     "wilcoxon_p": p_c, "dNI": d_ni.mean(), "sig_wins_minus_losses": wins}
                )
    pd.set_option("display.width", 250)
    print(f"{args.benchmark}, seeds={args.seeds} ({len(keep)})")
    print(pd.DataFrame(rows).sort_values("NI", ascending=False).round(3).to_string(index=False))
    print()
    print(pd.DataFrame(tests).round(4).to_string(index=False))
    return 0


def _pad(curves: np.ndarray, n: int) -> np.ndarray:
    """Curves cut at 50% padded with their last value; seed_auc reads no padded column."""
    width = curves.shape[1]
    assert width >= int(np.rint(0.5 * n))
    return np.concatenate([curves, np.repeat(curves[:, -1:], n - width, axis=1)], axis=1)


if __name__ == "__main__":
    raise SystemExit(main())
