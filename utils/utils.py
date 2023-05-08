from typing import Dict
import os

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import gc
import sys
sys.path.append('../')


def load_config(path: str,
                default_arguments: str='configs/base/default_arguments.yaml') -> Dict:
    
    custom_config = load(open(path, 'r'), Loader=Loader)
    
    # Load default config
    if default_arguments:
        assert os.path.exists(default_arguments), f"Can't find base config {default_arguments}"
        default_config = load(open(default_arguments, 'r'), Loader=Loader)
        for key, value in custom_config.items():
            default_config[key] = value
        return default_config
    gc.collect()
    return custom_config

import gdown

def load_data(dataset):
    datasets = {
    "mind_small" : "https://drive.google.com/drive/folders/1vtifsc492X-JilZto-38LIPsCMzdETKc?usp=share_link",
    "mind_large" : "https://drive.google.com/drive/folders/1HNvaSKGJ_x3twR5w9rCAhV25GohqI0dC?usp=share_link",
    "ttrs" : ""
    }
    output = f'data/preprocessed/{dataset}'
    assert dataset in datasets.keys(), f"Specify the correct dataset. {dataset} is not correct one."
    url = datasets[dataset]
    if not os.path.exists(os.path.join(os.getcwd(), output)):
        print("Creating dir data")
        os.makedirs(os.path.join(os.getcwd(), output))
        print("Loading data")
        gdown.download_folder(url, output=os.path.join(os.getcwd(), output))
    gc.collect()
    print(f"Check {os.path.join(os.getcwd(), output)}")