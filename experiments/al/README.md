# Active-learning loop for the production model

The v1 benchmark (`frozen/v1`) is frozen. Everything here only feeds the production models (user decision 2026-10-16).
Round 1 (`s1r1`) is part of v1. Round 2 (`s1r2`, 18,411 frames, TDDFT done) is the first production round; rounds 3 and 4
follow (user, 2026-10-18), and the loop stops earlier if the criterion below is met.

## Convergence criterion (trajectories)
`experiments/al/convergence.py` on the validation trajectories (`validation_trajectories.json`: styrene, azobenzene,
cyclohexenone, hexatriene, cytosine, benzaldehyde, cinnamaldehyde; TD-wB97X-D S1 dynamics, 500 fs). These molecules,
the paper's test trajectories (acrolein, uracil, trans-stilbene) and the extra test trajectories
(`test_trajectories_extra.json`: HBDI, DMABN) never enter train/val (`build_dataset.py` removes them).
Stop when (1) the models' own S1 dynamics from the validation minima are stable (finite, no bond change, drift < 0.1 eV)
and (2) the mean S1-gap and S0-rise MAE along the validation trajectories are <= 0.2 eV or improve by < 10 % over the
previous round; at the latest after round 4. Additionally reported per round: the pre-label error of the new frames
(the sampling model's prediction vs the new TDDFT labels, by double-bond twist).

## One round (N = 3, 4; tag of the dataset that contains rounds 2..N-1)
1. Dataset: `python experiments/al/build_dataset.py --rounds s1r2 [s1r3 ...] --tag vN-1` (local or dev-cpu job; ~5 min)
   -> `data/wb97_prod_<tag>.pkl`, `experiments/al/split_prod_<tag>.json`, `info_prod_<tag>.json`. Copy to HoreKa2.
2. Sampling models: `python experiments/al/make_specs.py sampler --tag <tag> --init RUN_A RUN_B` (round 3: model B
   runs `experiments/conj/runs_B/B_leh_s{0,1}`; later rounds: the previous sampling models), then on HoreKa2
   `.venv/bin/python experiments/run_queue.py experiments/al/spec_sampler_<tag>.yaml --dry` and
   `sbatch hpc/hk_train_chain.sh experiments/al/runs/sampler_<tag>_s{0,1}/job.yaml` (~8-10 h).
3. Criterion: `sbatch hpc/hk_cmd_cpu.sh ".venv/bin/python experiments/al/convergence.py --tag rN-1 experiments/al/runs/sampler_<tag>_s0 experiments/al/runs/sampler_<tag>_s1 --prev experiments/al/convergence_rN-2.json"`
   (GPU node if slow). Stop here if converged.
4. Sampling: `sbatch hpc/hk_al_sampling.sh s1rN experiments/al/runs/sampler_<tag>_s0 experiments/al/runs/sampler_<tag>_s1 0.05` (~2 h)
   -> `data/wb97/tasks/s1rN.pkl`; copy to bwUniCluster.
5. TDDFT: on bwUniCluster `bash hpc/al_tddft_submit.sh s1rN` (~6-7k core-h, 1-2 days incl. queue); copy
   `data/wb97/s1rN/*.npz` back (raw archives stay there).
6. Pre-label error of round N: `python experiments/al/prelabel_stats.py s1rN`.

Steps 3-4 can be chained to the training jobs: `sbatch --dependency=afterany:J1:J2 hpc/hk_al_round.sh rN-1 RUN_A RUN_B
s1rN experiments/al/convergence_rN-2.json` checks that both runs finished, evaluates the criterion (cpu partition, 4 h)
and submits the sampling only if the criterion is not met. Round 3 used `convergence_r1_benchmark.json` (the frozen
benchmark LEH model) as previous round.

## Production models
After the last round: build the final dataset, `python experiments/al/make_specs.py production --tag <final>`, train
(LEH 2 seeds, softmin 1 seed; bounded LEH blocks, `bound_on 12`, `bound_pair 5` eV), then evaluate the frozen test
sets, the new `test_*_exc{N}` sets and the test trajectories (`experiments/md/eval_models.py`) for the paper figure.
