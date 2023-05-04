#!/usr/bin/env python
# coding: utf-8

import argparse
import importlib
import pandas as pd
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
import monai
from monai.data import CSVDataset, CacheDataset, DataLoader
from monai.inferers import sliding_window_inference

from torchmetrics import MetricCollection, Dice, Accuracy, JaccardIndex
from torchmetrics.classification import BinaryFBetaScore



class VesuvisDataModule(pl.LightningDataModule):

    def __init__(self, params, train_transforms, val_transforms):
        super().__init__()

        self.params = params
        self.train_transform = train_transforms
        self.val_transform = val_transforms

        self.df = pd.read_csv(params["train_data_csv_path"])
        
    def _load_transforms(self, predict: bool = False):
        return [
            monai.transforms.LoadImaged(
                keys="volume_npy",
            ),
            monai.transforms.LoadImaged(
                keys=("mask_npy", "label_npy") if not predict else "mask_npy",
                ensure_channel_first=True,
            ),
        ]

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_val_df = self.df[self.df.stage == "train"].reset_index(drop=True)

            train_df = train_val_df[train_val_df.fragment_id != int(self.params["val_fragment_id"])].reset_index(
                drop=True
            )

            val_df = train_val_df[train_val_df.fragment_id == int(self.params["val_fragment_id"])].reset_index(drop=True)

            self.train_dataset = self._dataset(train_df, self._load_transforms(), self.train_transform)
            self.val_dataset = self._dataset(val_df, self._load_transforms(), self.val_transform)

            print(f"# train: {len(self.train_dataset)}")
            print(f"# val: {len(self.val_dataset)}")


    def _dataset(self, df, load_transform, transform):
        return CacheDataset(data=CSVDataset(src=df,
                                            transform=monai.transforms.Compose(load_transform)),
                            transform=transform,
                            cache_rate=1.0, runtime_cache="processes", copy_cache=False)

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def _dataloader(self, dataset, train=False):
        return DataLoader(
            dataset,
            batch_size=self.params["batch_size"],
            shuffle=train,
            num_workers=self.params["num_workers"],
            pin_memory=True,
            drop_last=train,
        )


class VesuvisModule(pl.LightningModule):

    def __init__(self, params, model, loss_function,):
        super().__init__()

        self.save_hyperparameters(ignore=["model", "loss_function"])

        self.params = params
        self.model = model
        self.loss = loss_function

        self.metrics = self._init_metrics()

    def _init_metrics(self):
        metric_collection = MetricCollection(
            {
                "dice": Dice(),
                "fbeta": BinaryFBetaScore(beta=0.5),
                "accuracy": Accuracy(task='binary'),
                "IoU": JaccardIndex(task='binary')
            }
        )

        return torch.nn.ModuleDict(
            {
                "train_metrics": metric_collection.clone(prefix="train/"),
                "val_metrics": metric_collection.clone(prefix="val/"),
            }
        )

    def configure_optimizers(self):
        optimizer = self.params["optimizer"](self.model.parameters(), **self.params["optimizer_params"])
       
        if self.params["scheduler"]:
            if self.params["scheduler"].__name__ == "OneCycleLR":
                scheduler = self.params["scheduler"](optimizer, total_steps=self.trainer.estimated_stepping_batches,
                                                     **self.params["scheduler_params"])
                scheduler = {"scheduler": scheduler, "interval" : "step"}
            
            else:
                scheduler = self.params["scheduler"](optimizer, **self.params["scheduler_params"])
                
            optimizer_dict = {'optimizer': optimizer,
                              'lr_scheduler': scheduler}

            return optimizer_dict
        
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def _shared_step(self, batch, stage):
        outputs, labels, masks = self._forward_pass(batch, stage)

        loss = self.loss(outputs, labels, masks)

        self.metrics[f"{stage}_metrics"](outputs, labels)

        self._log(loss, stage, batch_size=len(outputs))

        return loss

    def _forward_pass(self, batch, stage):
        volumes = batch["volume_npy"].as_tensor()
        masks = batch["mask_npy"].as_tensor()

        if stage == "train":
            outputs = self(volumes)
        elif stage == "val":
            outputs = sliding_window_inference(
                inputs=volumes,
                roi_size=self.params["patch_size"],
                sw_batch_size=self.params["sw_batch_size"],
                predictor=self,
                overlap=0.5,
                mode="gaussian",
            )

        try:
            labels = batch["label_npy"].as_tensor().long()
            return outputs, labels, masks
        except KeyError:
            return outputs, masks

    def _log(self, loss, stage, batch_size):
        self.log(f"{stage}/loss", loss, batch_size=batch_size)
        self.log_dict(self.metrics[f"{stage}_metrics"], batch_size=batch_size)


def train(params_path, params):
    monai.utils.set_determinism(params.PARAMS["seed"])
    pl.seed_everything(params.PARAMS["seed"], workers=True)

    data_module = VesuvisDataModule(params.PARAMS,
                                    params.get_train_transforms(),
                                    params.get_val_transforms())

    module = VesuvisModule(params.PARAMS, params.model, params.loss_function)

    tb_logger = TensorBoardLogger(save_dir='./logs', name=params.PARAMS["exp_name"], default_hp_metric=False)

    model_checkpoint_dir = os.path.join('./logs', params.PARAMS["exp_name"], 'version_' + str(tb_logger.version), 'models')
    checkpoint_callback = ModelCheckpoint(verbose=True,
                                          save_top_k=3,
                                          monitor=params.PARAMS["ckpt_monitor"],
                                          mode='max',
                                          dirpath=model_checkpoint_dir,
                                          filename='epoch={epoch:02d}-{step}-dice={val/dice:.5f}',
                                          auto_insert_metric_name=False)

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        accelerator=params.PARAMS["accelerator"],
        benchmark=True,
        check_val_every_n_epoch=1,
        devices=params.PARAMS["devices"],
        callbacks=[LearningRateMonitor(logging_interval='step'),
                   checkpoint_callback],
        logger=tb_logger,
        log_every_n_steps=1,
        max_epochs=params.PARAMS["epochs"],
        accumulate_grad_batches=max(1, params.PARAMS["acc_batch_size"]//params.PARAMS["batch_size"]),
        precision=params.PARAMS["precision"],
        strategy="ddp" if params.PARAMS["devices"] > 1 else 'auto',
    )

    with open(params_path) as f:
        write_params = f.read()
    trainer.logger.experiment.add_text(params_path, write_params)

    trainer.fit(module, datamodule=data_module)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get all command line arguments.')
    parser.add_argument('params', type=str, help='Path to parameters py file')
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("params", args.params)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    train(args.params, params)