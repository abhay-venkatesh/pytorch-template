from tqdm import tqdm
from lib.simple_ml_logger.logger import Logger
import torch
from pathlib import Path


class Trainer:
    def __init__(self, experiment):
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')
        self.logger = Logger
        self.experiment = experiment

    def train(self):
        self._setup_model()
        self._setup_optimizer()
        self._load_checkpoint()
        for epoch in tqdm(
                range(self.start_epochs, self.experiment.config["epochs"])):

            total_loss = 0
            self.model.train()
            for X, Y in tqdm(self.train_loader):
                X, Y = X.to(self.device), Y.to(self.device)
                total_loss += self._step(X, Y)
            avg_loss = total_loss / len(self.train_loader)
            self.logger.log("epoch", "avg_loss", epoch, avg_loss)

            val_metric = self.validate()
            self.logger.log("epoch", "val_metric", epoch, val_metric)

    def _step(self, X, Y):
        raise NotImplementedError

    def validate(self):
        """
            returns:
                val_metric: your favorite evaluation metric
        """
        raise NotImplementedError

    def _load_checkpoint(self):
        self.start_epochs = 0
        if self.experiment.config["checkpoint path"]:
            self.start_epochs = int(Path(self.checkpoint_path).stem)
            self.model.load_state_dict(torch.load(self.checkpoint_path))

    def _setup_model(self):
        raise NotImplementedError

    def _setup_optimizer(self):
        raise NotImplementedError