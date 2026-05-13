## Installation
``` Bash
conda create -n env_name python=3.9
pip install -r requirements.txt
```

## Environment Installation
Install the ``MetaDrive`` environment via
```
pip install git+https://github.com/HenryLHH/metadrive_clean.git@main
```

## Main results
Run
``` Bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
python train.py --config train_config.py:cbf --env_id 30 
```
where ``env_id`` serves as an index for the [list of environments](https://github.com/ZhengYinan-AIR/FISOR/tree/metadrive_imitation/env/env_list.py).
