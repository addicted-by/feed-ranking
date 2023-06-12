from recbole.quick_start import run_recbole
import sys
import os
sys.path.append('../')
sys.path.append(os.getcwd())
from utils.utils import load_config, update_markdown

if __name__ == '__main__':
    config_dict = load_config('configs/sequential/gru4rec.yaml')
    run_recbole(
        model='GRU4Rec',
        dataset='mind_small',
        config_dict=config_dict
    )
    update_markdown('GRU4Rec', 'mind_small', 'GRU baseline')