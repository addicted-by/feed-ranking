from recbole.quick_start import run_recbole
import sys
import os
sys.path.append('../')
sys.path.append(os.getcwd())
from utils.utils import load_config

if __name__ == '__main__':
    config_dict = load_config('configs/general/bpr.yaml')
    run_recbole(
        model='BPR',
        dataset='mind_small',
        config_dict=config_dict
    )