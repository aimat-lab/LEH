"""One X-MACE baseline run on our splits: export (exmlip env) -> train (authors' run_train.py, X-MACE env) ->
predict into our format. Run with .venv_xmace. Training stops at --hours; then the model is finalised from the
latest checkpoint (MACE --restart_latest with max_num_epochs = last epoch + 1).

Defaults follow the X-MACE README (AutoencoderExcitedMACE, lr 1e-4, batch 10, EMA 0.99, energy/forces weight 100,
float32, 2 interactions, correlation 3, 8 radial functions, r_max 5); --channels selects 128 (README) or 32 (ours).
"""
import argparse
import glob
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["wb97", "spectra"], required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--run", required=True, help="our run dir, e.g. experiments/wb97/runs/xmace")
ap.add_argument("--n_states", type=int, required=True)
ap.add_argument("--channels", type=int, default=128)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--hours", type=float, default=7.0)
ap.add_argument("--device", default="cuda")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint up to --epochs")
a = ap.parse_args()
run = ROOT / a.run
exp = run / "export"
run.mkdir(parents=True, exist_ok=True)
if not (exp / "index.json").exists():
    subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "experiments/baselines/xmace_export.py"), "--mode", a.mode,
                    "--data", str(ROOT / a.data), "--split", str(ROOT / a.split), "--out", str(exp),
                    "--n_states", str(a.n_states)], check=True)
name = "xmace"
fw = "100.0" if a.mode == "wb97" else "0.0"
base = [sys.executable, str(ROOT / "external/X-MACE/scripts/run_train.py"), f"--name={name}",
        f"--train_file={exp / 'train.xyz'}", f"--valid_file={exp / 'valid.xyz'}", f"--seed={a.seed}", "--E0s=average",
        "--model=AutoencoderExcitedMACE", "--r_max=5.0", "--batch_size=10", f"--n_energies={a.n_states}",
        "--correlation=3", "--ema", "--ema_decay=0.99", "--lr=0.0001", "--default_dtype=float32", f"--device={a.device}",
        f"--hidden_irreps={a.channels}x0e + {a.channels}x1o", f"--MLP_irreps={a.channels}x0e", "--num_radial_basis=8",
        "--num_interactions=2", "--energy_weight=100.0", f"--forces_weight={fw}", "--error_table=EnergyNacsDipoleMAE",
        f"--model_dir={run}", f"--log_dir={run / 'logs'}", f"--checkpoints_dir={run / 'checkpoints'}",
        f"--results_dir={run / 'results'}"]
log = open(run / "train_xmace.log", "a")
t0 = time.time()
model = run / f"{name}.model"


def latest_epoch():
    ck = glob.glob(str(run / "checkpoints" / "*_epoch-*.pt"))
    return max((int(re.search(r"epoch-(\d+)", p).group(1)) for p in ck), default=-1)


if not model.exists() or a.resume:
    start = latest_epoch()
    if start + 1 < a.epochs:
        extra = ["--restart_latest"] if start >= 0 else []
        print(f"training from epoch {start + 1} towards {a.epochs}", file=log, flush=True)
        try:
            subprocess.run(base + [f"--max_num_epochs={a.epochs}"] + extra, cwd=run, stdout=log,
                           stderr=subprocess.STDOUT, timeout=a.hours * 3600, check=True)
        except subprocess.TimeoutExpired:
            last = latest_epoch()
            print(f"time limit: finalising from checkpoint epoch {last}", file=log, flush=True)
            subprocess.run(base + [f"--max_num_epochs={last + 1}", "--restart_latest"], cwd=run, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
(run / "epochs_done.txt").write_text(str(latest_epoch() + 1))
subprocess.run([sys.executable, str(ROOT / "experiments/baselines/xmace_predict.py"), str(model), str(exp), str(run),
                "--device", a.device], check=True)
print("done", (time.time() - t0) / 3600, "h", file=log, flush=True)
