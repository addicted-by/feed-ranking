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
    
    assert os.path.exists(path), f"Bad specified path {path}"
    custom_config = load(open(path, 'r'), Loader=Loader)
    
    # Load default config
    if default_arguments:
        assert os.path.exists(default_arguments), f"Can't find base config {default_arguments}"
        default_config = load(open(default_arguments, 'r'), Loader=Loader)
        if custom_config:
            for key, value in custom_config.items():
                default_config[key] = value
        return default_config
    gc.collect()
    return custom_config

import re
import glob

def extract_metrics_from_log(log_file):
    with open(log_file, 'r') as file:
        log_content = file.read()
    metrics = re.findall(r"metrics = \[([^]]+)\]", log_content)[0].split(', ')
    metrics = [str(metric.lower())[1:-1] for metric in metrics]
    results_dict = {}
    for metric in metrics:
        metric_dict = {}
        pattern = r'{}@(\d+) : ([\d.]+)'.format(metric)
        metric_matches = re.findall(pattern, log_content)
        for key, value in metric_matches:
            if key in metric_dict:
                metric_dict[key].append(float(value))
            else:
                metric_dict[key] = [float(value)]
        results_dict[metric] = metric_dict
    return results_dict


def get_best_metrics(log_file):
    metrics = extract_metrics_from_log(log_file)
    results = {}
    for metric, ks in metrics.items():
        for k, values in ks.items():
            results[metric + "@" + k] = max(values)

    return results

def get_last_log(model_type, dataset):
    list_of_files = glob.glob(f'log/{model_type}/{model_type}-{dataset}-*')
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def update_markdown(model_type, dataset, name, log_file: str=None):
    if log_file:
        data = {name: get_best_metrics(log_file)}
    else:
        data = {name: get_best_metrics(get_last_log(model_type, dataset))}
    metrics = ["precision@1", "precision@5", "precision@10",
           "recall@1", "recall@5", "recall@10",
           "map@1", "map@5", "map@10",
           "ndcg@1", "ndcg@5", "ndcg@10"]

    with open("README.md", "r") as file:
        markdown_content = file.read()

    table_regex = r"\| Model .*? \|.*?\|\s*\n([\s\S]*?)\n\n"
    table_match = re.search(table_regex, markdown_content, re.MULTILINE)

    if table_match:
        existing_table = table_match.group(1)

        table_data = [line.split("|")[1:-1] for line in existing_table.split("\n") if line.strip()]
        new_row = [name]
        new_row.extend([str(data[name][metric]) for metric in metrics])
        table_data.append(new_row)
        updated_table_content = "\n".join("| " + " | ".join(row) + " |" for row in table_data)

        header = "| Model | " + " | ".join(metrics) + " |\n"
        updated_markdown_content = re.sub(table_regex, f"{header}{updated_table_content}\n\n", markdown_content, flags=re.MULTILINE)

        with open("README.md", "w") as file:
            file.write(updated_markdown_content)
    else:
        print("No table found in the Markdown file.")

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