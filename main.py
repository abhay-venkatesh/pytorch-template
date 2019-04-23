from lib.trainers import *  # noqa F401
from lib.utils.experiment import Experiment
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    args = parser.parse_args()

    experiment = Experiment(args.config_file)
    Trainer = globals()[experiment.config["trainer"]]
    Trainer(experiment).train()