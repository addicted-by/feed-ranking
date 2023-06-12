from utils.utils import load_data
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Loading data for Recommender Systems")
    parser.add_argument("--dataset", help="Dataset name")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    dataset_name = args.dataset
    load_data(dataset_name)