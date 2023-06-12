from utils.utils import update_markdown
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Update README with corresponding log path")
    parser.add_argument("--log", help="log path")
    parser.add_argument("--name", help="Row name")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    log = args.log
    splitted = os.path.basename(log).split('-')
    model_type = splitted[0]
    dataset = splitted[1]
    update_markdown(model_type=model_type, dataset=dataset, name=args.name, log_file=log)