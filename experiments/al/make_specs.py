"""Run specs of the active-learning loop (procedure: experiments/al/README.md).

sampler: the two sampling models of the next round, warm-started (train.py init_from) from the previous models and
  fine-tuned on the production dataset of this tag (60k updates, lr 1e-3, cosine), with bounded LEH blocks.
production: the final models, trained from scratch on the final production dataset with the benchmark settings (128 ch
  L1, b16, lr 3e-3, cosine, EMA 0.99, w_e0 100, 300k updates) and bounded LEH blocks: LEH (2 seeds, also the committee
  for uncertainty) and Direct-softmin (1 seed, for the LEH-vs-softmin comparison along the trajectories).
Usage: python experiments/al/make_specs.py sampler --tag v2 --init experiments/conj/runs_B/B_leh_s0 experiments/conj/runs_B/B_leh_s1
       python experiments/al/make_specs.py production --tag v4
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BOUNDS = {"bound_on": 12.0, "bound_pair": 5.0}  # eV; 2x the largest block elements on normal test geometries
BASE = yaml.safe_load(open(ROOT / "experiments/wb97/spec_main.yaml"))["base"]

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["sampler", "production"])
ap.add_argument("--tag", required=True)
ap.add_argument("--init", nargs=2, default=None)
a = ap.parse_args()
base = dict(BASE, data=f"data/wb97_prod_{a.tag}.pkl", split=f"experiments/al/split_prod_{a.tag}.json")
leh = dict(head="latent", head_kwargs=dict(k=1, **BOUNDS))
if a.kind == "sampler":
    assert a.init, "--init RUN_A RUN_B required"
    base.update(max_updates=60000, lr=0.001, warmup_updates=500, max_hours=20.0)
    runs = [dict(name=f"sampler_{a.tag}_s{k}", seed=k, init_from=src, **leh) for k, src in enumerate(a.init)]
    out = dict(outdir="experiments/al/runs", base=base, runs=runs)
    hdr = f"# Sampling models for the round after dataset {a.tag} (experiments/al/make_specs.py; README.md)\n"
else:
    base.update(max_updates=300000, max_hours=80.0)
    runs = [dict(name=f"prod_leh_{a.tag}_s{s}", seed=s, **leh) for s in (0, 1)] + \
           [dict(name=f"prod_softmin_{a.tag}_s0", seed=0, head="softmin")]
    out = dict(outdir="experiments/al/runs", base=base, runs=runs)
    hdr = f"# Production models on dataset {a.tag} (experiments/al/make_specs.py; README.md)\n"
f = ROOT / f"experiments/al/spec_{a.kind}_{a.tag}.yaml"
f.write_text(hdr + yaml.safe_dump(out, sort_keys=False, default_flow_style=None))
print("written", f)
