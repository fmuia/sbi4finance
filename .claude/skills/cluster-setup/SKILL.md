---
name: cluster-setup
description: >
  Set up and use Brown's Oscar cluster for sbi4finance: check ssh access,
  discover which SLURM lanes the user holds, install pixi on Oscar, and
  submit the GPU training job. Use when the user asks to "set up the cluster",
  "configure Oscar", "what resources do I have", "why is my job queueing",
  "train on the cluster", or runs a job there for the first time.
---

# Using Oscar for sbi4finance

Everything in `cluster/` is lab-wide. What differs per person is which SLURM
associations they hold, and therefore how much they can run at once. Discovery
turns that into numbers; this skill walks the user through it with agreement at
each step rather than guesses.

**Do not answer resource questions from memory or from this file.** The
numbers below are examples from one account. Discovery is the source of truth.

## Step 0: ssh access, once per 12 h window

Oscar needs password + Duo. `~/.ssh/config` should hold this block so one
interactive login serves every later command (and the discovery script):

```
Host oscar
  HostName ssh.ccv.brown.edu
  User <brown username>
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 12h
  ServerAliveInterval 15
  ServerAliveCountMax 20
```

Then, in a terminal the user controls (an agent cannot answer Duo):

```bash
ssh oscar 'echo CONNECTED; hostname'
```

If an agent's `ssh -o BatchMode=yes oscar true` says `Permission denied
(keyboard-interactive)`, the control socket is not open. Ask the user to run
the line above, then retry. Do not loop on it.

## Step 1: discover the lanes

```bash
pixi run python lab/cluster/discover_cluster.py --ssh-host oscar
```

It writes `lab/cluster/oscar.local.json` (gitignored) and prints one line per lane.
A lane is one (account, partition, QOS) triple; its cap is `MaxTRESPU` on the
QOS, applied per QOS, so lanes are independent budgets. **Show the user the
summary and confirm it looks right before continuing.** A surprising lane
usually means their allocation changed.

What one lab account looked like on 2026-08-09 (example, re-discover):

| account | partition | QOS | priority | cap |
|---|---|---|---|---|
| carney-mjfrank-condo2 | batch | carney-condo2 | 10000 | 208 cores, 2.5 TB |
| carney-mjfrank-condo2 | gpu-he | carney-gcondo2 | 10000 | 160 cores, 75 GPUs |
| default | batch | normal | 0 | 64 cores |
| default | gpu | norm-gpu | 0 | 12 cores, 2 GPUs |

`gpu-he` nodes carry 8 GPUs each (L40S, H100, B200, RTX Pro 6000 classes), so a
1-GPU request packs well and schedules faster than a whole node. Condo lanes
carry priority 10000; `default` lanes are priority 0 and only worth it when the
condo is busy. Check before promising a speedup:

```bash
ssh oscar 'sinfo -h -p gpu-he -o "%t %D" | sort'   # idle vs alloc vs mix
ssh oscar 'squeue -h -u $USER | wc -l'             # what they already have queued
```

## Step 2: agree the mapping

1. **Training** (`cluster/train.sbatch`): one GPU on the highest-priority GPU
   lane. One job cannot span lanes, so there is nothing to split. 8 cores and
   64 GB are generous for the FusionTransformer + CouplingFlow; measure
   `sacct -j <id> --format=JobID,Elapsed,MaxRSS` once and trim.
2. **Simulation** (future array jobs): the simulator does 100k paths in about
   a second on one core, so simulation does not need the cluster unless the
   model changes. Use the best CPU lane, one core per task, when it does.
3. **Where output goes**: never the home directory (small, often nearly full).
   Clone the repo under the lab's data volume and check it with `checkquota`.
4. **Wall time** is a quoted string in the sbatch file on purpose. Ask for
   what the run needs, not the maximum: the full 100k x 60-epoch run took
   about 10 min on a 2022 GPU.

## Step 3: pixi on Oscar

pixi is user-space; no module needed. The package cache must not land in the
home quota:

```bash
ssh oscar
curl -fsSL https://pixi.sh/install.sh | sh
echo 'export PIXI_CACHE_DIR=/oscar/data/<lab>/$USER/pixi-cache' >> ~/.bashrc   # confirm the path with checkquota
cd /oscar/data/<lab>/$USER && git clone <repo> sbi4finance && cd sbi4finance
pixi install -e gpu          # linux-64 CUDA env; login nodes have internet
pixi run -e gpu python -c "import jax; print(jax.devices())"   # prints CpuDevice on a login node, fine
```

## Step 4: submit, then watch

```bash
sbatch cluster/train.sbatch                                   # lab defaults
sbatch --account=<acct> --partition=<part> cluster/train.sbatch --n-train 20000 --epochs 10
squeue -u $USER
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS
tail -f sbi4finance-train-<job_id>.out
```

`sbatch` flags override `#SBATCH` lines; arguments after the script go to
`scripts/train_univariate.py`. The job writes to `checkpoints/univariate-retrained/`;
rename to `checkpoints/univariate/` only after the notebook's recovery and
calibration plots look right with it.

## Rules

- **Never commit a personal value.** `oscar.local.json` is gitignored and
  generated. Account names of lab condos are fine in `train.sbatch`; usernames,
  paths under `/oscar/data`, and tokens are not.
- **Never hand-edit the generated file**; re-run discovery.
- **Re-run discovery** when allocations change and after any "why is this
  queueing?" question.
- Do not submit on the user's behalf without showing the exact `sbatch` line.
