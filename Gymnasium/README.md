## Installation
``` Bash
conda create -n env_name python=3.9
pip install -r requirements.txt
```

## Main results
Run
``` Bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
python train.py --config train_config.py:cbf --env_id 19
```
where ``env_id`` serves as an index for the given in [list of environments](env/env_list.py)