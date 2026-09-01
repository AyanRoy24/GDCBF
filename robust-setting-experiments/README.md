Robust barrier update under dataset corruption (Figure 2 / Appendix B.3).

70% of high-cost trajectories are relabelled as safe, then the deterministic
Smooth CBF (symmetric L1) is compared against the robust variant, which uses a
percentile loss at `tau = 0.6`.

## Installation
``` Bash
conda create -n env_name python=3.9
pip install -r requirements.txt
```

## Environment Installation
For MetaDrive tasks (``env_id >= 30``) install
```
pip install git+https://github.com/HenryLHH/metadrive_clean.git@main
```

## Main results
Run
``` Bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
python train.py --config train_config.py:r --env_id 34 --mode 2   # robust
python train.py --config train_config.py:r --env_id 34 --mode 1   # deterministic
```
where ``env_id`` serves as an index for the [list of environments](env/env_list.py)
