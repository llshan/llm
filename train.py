import os
import math
import yaml
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data_module import WikiLMDataModule
from lit_gpt import LitGPT2


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    dm = WikiLMDataModule(
        tokenizer_name=cfg["tokenizer_name"],
        train_path=cfg["data"]["train_path"],
        val_path=cfg["data"]["val_path"],
        block_size=cfg["data"]["block_size"],
        train_batch_size=cfg["data"]["train_batch_size"],
        val_batch_size=cfg["data"]["val_batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    dm.setup()
    num_train_examples = len(dm.dataset["train"])
    block_size = cfg["data"]["block_size"]
    num_train_tokens = num_train_examples * block_size

    tokens_per_step = block_size * cfg["data"]["train_batch_size"]
    steps_per_epoch = max(1, num_train_tokens // tokens_per_step)
    total_steps = steps_per_epoch * cfg["training"]["max_epochs"]

    model = LitGPT2(
        model_size=cfg["model_size"],
        tokenizer_name=cfg["tokenizer_name"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_steps=cfg["training"]["warmup_steps"],
        total_steps=total_steps,
    )

    logger = TensorBoardLogger(
        save_dir=cfg["training"]["default_root_dir"],
        name=cfg["experiment_name"],
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(
            cfg["training"]["default_root_dir"], cfg["experiment_name"]
        ),
        filename="{epoch}-{step}-{val_loss:.3f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator=cfg["training"]["accelerator"],
        devices=cfg["training"]["devices"],
        max_epochs=cfg["training"]["max_epochs"],
        precision=cfg["training"]["precision"],
        logger=logger,
        callbacks=[checkpoint_cb],
        log_every_n_steps=cfg["training"]["log_every_n_steps"],
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

