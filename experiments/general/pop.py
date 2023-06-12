from recbole.quick_start import run_recbole
import sys
import os
sys.path.append('../')
sys.path.append(os.getcwd())
from utils.utils import load_config

if __name__ == '__main__':
    config_path = 'configs/general/pop.yaml'
    config_dict = load_config(config_path)
    run_recbole(
        model='Pop',
        dataset='mind_small',
        config_dict=config_dict
    )