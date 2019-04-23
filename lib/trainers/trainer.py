from lib.simple_ml_logger.logger import Logger
import torch


class Trainer:
    def __init__(self, experiment):
        self.experiment = experiment
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')
        self.logger = Logger

    def train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError
