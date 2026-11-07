"""Add the pretrained OMNI-P2x fine-tuned with the authors' protocol (full training set, model A data) to
experiments/conj/model_B_vs_A.json as "A OMNI-P2x ft", with the metrics of analyze_B.py on the conjugated test sets.
Input: experiments/wb97/runs_lc/suball_omnift_conj/predictions.pkl (copied from HoreKa2).
"""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/conj"))
from analyze_B import CONJ, metrics  # noqa

P = pickle.load(open(ROOT / "experiments/wb97/runs_lc/suball_omnift_conj/predictions.pkl", "rb"))
f = ROOT / "experiments/conj/model_B_vs_A.json"
R = json.load(open(f))
R["A OMNI-P2x ft"] = {s: {k: (float(v), 0.0, 1) for k, v in metrics(P[s]).items()} for s in CONJ if P.get(s)}
json.dump(R, open(f, "w"), indent=1)
print({s: round(R["A OMNI-P2x ft"][s]["S1-3 MAE"][0], 3) for s in R["A OMNI-P2x ft"]})
