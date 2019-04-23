from lib.utils.experiment import Experiment
import argparse
import importlib
import inflection

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    args = parser.parse_args()

    experiment = Experiment(args.config_file)

    trainer = importlib.import_module(("lib.trainers.{}").format(
        inflection.underscore(experiment.config["trainer"])))
    Trainer = getattr(trainer, experiment.config["trainer"])
    Trainer(experiment).train()