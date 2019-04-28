from lib.datasets.coco_stuff import COCOStuff
from lib.models.segnet import get_model
from lib.utils.functional import cross_entropy2d, get_iou
from lib.agents.trainer import Trainer
from pathlib import Path
from statistics import mean
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


class COCOStuffTrainer(Trainer):
    def run(self):
        trainset = COCOStuff(Path(self.config["dataset path"], "train"))
        train_loader = DataLoader(
            dataset=trainset,
            batch_size=self.config["batch size"],
            shuffle=True)

        valset = COCOStuff(Path(self.config["dataset path"], "val"))
        val_loader = DataLoader(
            dataset=valset, batch_size=self.config["batch size"])

        model = get_model(n_classes=trainset.N_CLASSES).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config["learning rate"])

        for epoch in tqdm(range(self.config["epochs"])):

            model.train()
            total_loss = 0
            for X, Y in tqdm(train_loader):
                X, Y = X.to(self.device), Y.long().to(self.device)
                Y_ = model(X)
                loss = cross_entropy2d(Y_, Y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            avg_loss = total_loss / len(train_loader)
            self.logger.log("epoch", epoch, "loss", avg_loss)

            model.eval()
            ious = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels[0].long()
                    labels = labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    iou = get_iou(predicted, labels)
                    ious.append(iou.item())

            mean_iou = mean(ious)
            self.logger.log("epoch", epoch, "iou", mean_iou)

            self.logger.graph()
