

from __future__ import annotations

import datetime
import itertools
import os
import pickle
import torch
import torch.optim as optim
from prettytable import PrettyTable
from torch.utils.data import DataLoader

import wandb

from omni.isaac.lab.utils.io import dump_yaml

from fdm.model import EmpiricalNormalization, FDMModel
from omni.isaac.lab_tasks.utils import get_checkpoint_path

from .early_stopping import EarlyStopping
from .replay_buffer_cfg import ReplayBufferCfg
from .trainer_cfg import TrainerBaseCfg
from .trajectory_dataset import TrajectoryDataset


class Trainer:
    def __init__(
        self,
        cfg: TrainerBaseCfg,
        replay_buffer_cfg: ReplayBufferCfg,
        model: FDMModel,
        device: str = "cuda",
        eval: bool = False,
    ):
        self.cfg = cfg
        self.replay_buffer_cfg = replay_buffer_cfg
        self.device = device
        self.eval = eval
        self.model = model

        # init logging dir (also used for loading checkpoints)
        self._init_looging_dir()

        # resume model if specified
        if self.cfg.resume:
            resume_path = get_checkpoint_path(self.log_root_path, self.cfg.load_run, self.cfg.load_checkpoint)
            self.model.load(resume_path)
            print(f"[INFO]: Loaded model checkpoint from: {resume_path}")

            if self.eval:
                self.model.eval()

        # init dataset
        self.train_dataset: TrajectoryDataset = TrajectoryDataset(
            cfg=cfg, replay_buffer_cfg=self.replay_buffer_cfg, model_cfg=self.model.cfg, return_device=self.device
        )
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle_batch,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        if not self.eval:
            # resume certain encoders of the model
            if self.cfg.encoder_resume:
                for encoder_name, checkpoint_path in self.cfg.encoder_resume.items():
                    getattr(self.model, encoder_name).load_state_dict(torch.load(os.path.abspath(checkpoint_path)))
                    print(f"[INFO]: Loaded encoder {encoder_name} from: {checkpoint_path}")

                    # prevent that empiricial normalization is ever getting updated
                    if isinstance(getattr(self.model, encoder_name), EmpiricalNormalization):
                        getattr(self.model, encoder_name).until = 0

            # init validation dataset
            self.val_dataset: TrajectoryDataset = TrajectoryDataset(
                cfg=cfg, replay_buffer_cfg=self.replay_buffer_cfg, model_cfg=self.model.cfg, return_device=self.device
            )
            self.validation_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=self.cfg.shuffle_batch,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
            )

            # init optimizer
            if self.cfg.encoder_resume:
                model_param_generators = [
                    getattr(self.model, layer_name).parameters()
                    for layer_name in self.model.layer_parameters
                    if layer_name not in list(self.cfg.encoder_resume.keys())
                ]
                model_param = list(itertools.chain.from_iterable(model_param_generators))
            else:
                model_param = [*self.model.parameters()]
            self.optimizer = optim.Adam(model_param, lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
            # init learning rate scheduler and early stopping module
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5, verbose=True, min_lr=1e-6
            )
            if self.cfg.early_stopping:
                self.early_stopping = EarlyStopping(patience=3)

            # counter
            self.batch_counter = 0

            # best val loss
            self.best_val_loss = float("inf")

            # init logging
            self._init_logging()

        # load test dataset if given any
        if self.cfg.test_datasets:
            if isinstance(self.cfg.test_datasets, str):
                self.cfg.test_datasets = [self.cfg.test_datasets]

            self.test_datasets = {}
            for test_dataset_path in self.cfg.test_datasets:
                if not os.path.isfile(test_dataset_path):
                    print(f"[WARNING] Test Dataset {test_dataset_path} not found! Will proceed without!")
                    continue

                basename = os.path.splitext(os.path.split(test_dataset_path)[1])[0]
                with open(test_dataset_path, "rb") as test_dataset:
                    self.test_datasets[basename] = DataLoader(
                        pickle.load(test_dataset),
                        batch_size=self.cfg.batch_size,
                        shuffle=False,
                        num_workers=self.cfg.num_workers,
                        pin_memory=True,
                    )
        else:
            self.test_datasets = None

    """
    Operations
    """

    def train(self, collection_round: int | None = None):
        """Train the model."""
        # buffer variables for all epochs
        batch_number_train = len(self.dataloader)
        batch_number_val = len(self.validation_dataloader)
        epoch_mean_loss = 0
        epoch_mean_val_loss = 0
        if self.test_datasets:
            batch_number_test = sum([len(train_dataloader) for train_dataloader in self.test_datasets])
            epoch_mean_test_loss = 0

        # reset early stopping
        if self.cfg.early_stopping:
            self.early_stopping.reset()

        # log collection epoch
        if self.cfg.logging:
            wandb.log({"Collection Round": collection_round})

        for epoch in range(self.cfg.epochs):
            # buffer variable reset each epoch
            mean_loss = 0
            mean_val_loss = 0
            mean_test_loss = 0

            # training loop
            for inputs in self.dataloader:
                # inputs [state_history, obs_proprioceptive, obs_exteroceptive, actions, add_obs_extereoceptive, states, perfect_velocity_following_local_frame]
                loss, meta = self.model.update(
                    model_in=inputs[:5], optimizer=self.optimizer, target=inputs[5], eval_in=inputs[6:]
                )
                mean_loss += loss

                if self.cfg.logging:
                    wandb.log(meta, step=self.batch_counter)
                    self.batch_counter += 1
            # get mean training loss
            mean_loss = mean_loss / batch_number_train
            epoch_mean_loss += mean_loss

            # validation loop
            eval_meta = {}
            for inputs in self.validation_dataloader:
                loss, meta = self.model.evaluate(model_in=inputs[:5], target=inputs[5], eval_in=inputs[6:])
                mean_val_loss += loss
                if self.cfg.logging:
                    for key, value in meta.items():
                        eval_meta[key] = value + eval_meta.get(key, 0)
            # get mean validation loss
            mean_val_loss = mean_val_loss / batch_number_val
            epoch_mean_val_loss += mean_val_loss

            # testing loop (in the case any test datasets are given)
            if self.test_datasets:
                test_meta = {}
                for dataset_name, test_dataloader in self.test_datasets.items():
                    for inputs in test_dataloader:
                        loss, meta = self.model.evaluate(
                            model_in=inputs[:5],
                            target=inputs[5],
                            eval_in=inputs[6:],
                            mode="test",
                            suffix=f"_{dataset_name}",
                        )
                        mean_test_loss += loss
                        if self.cfg.logging:
                            for key, value in meta.items():
                                test_meta[key] = value + test_meta.get(key, 0)
                # get mean validation loss
                mean_test_loss = mean_test_loss / batch_number_test
                epoch_mean_test_loss += mean_test_loss

            if self.cfg.logging:
                # scale eval meta as logged per batch
                for key, value in eval_meta.items():
                    eval_meta[key] = value / batch_number_val

                wandb.log(
                    {
                        "train Loss [Epoch]": mean_loss,
                        "eval Loss [Epoch]": mean_val_loss,
                        "learning rate": self.optimizer.param_groups[0]["lr"],
                    }
                    | eval_meta,
                    step=self.batch_counter,
                )

                if self.test_datasets:
                    for key, value in test_meta.items():
                        test_meta[key] = value / batch_number_test

                    wandb.log({"test_loss [Epoch]": mean_test_loss} | test_meta, step=self.batch_counter)

            # print a description string
            desc_str = (
                f"Epoch {epoch}: lr:{self.optimizer.param_groups[0]['lr']:.6f}: - train loss:{mean_loss:.4f}"
                f" - val loss:{mean_val_loss:.4f}"
            )
            if self.test_datasets:
                desc_str = desc_str + f" -  test loss:{mean_test_loss:.4f}"
            desc_str = (
                f"Collection round {collection_round} - " + desc_str if collection_round is not None else desc_str
            )
            print(desc_str)

            # run learning rate scheduler
            old_lr = [group["lr"] for group in self.optimizer.param_groups]
            if not isinstance(collection_round, int) or collection_round > self.cfg.learning_rate_warmup - 1:
                self.scheduler.step(mean_val_loss)
            new_lr = [group["lr"] for group in self.optimizer.param_groups]

            # add pre-trained model weights to optimizer if lr rate decreased for the first time
            if old_lr != new_lr and self.cfg.encoder_resume and self.cfg.encoder_resume_add_to_optimizer:
                [
                    self.optimizer.add_param_group(
                        {"params": getattr(self.model, layer_name).parameters(), "lr": new_lr[0]}
                    )
                    for layer_name in self.model.layer_parameters
                    if layer_name in list(self.cfg.encoder_resume.keys())
                ]
                self.scheduler.min_lrs = self.scheduler.min_lrs + self.scheduler.min_lrs * len(
                    self.cfg.encoder_resume.keys()
                )

                # only add the encoders once
                self.cfg.encoder_resume_add_to_optimizer = False

            # update epoch counter for learning rate progress in model
            if hasattr(self.model, "_update_step"):
                self.model._update_step += 1

            # check for early stopping
            if self.cfg.early_stopping and self.early_stopping(mean_val_loss):
                print("[INFO] Early stopping")
                break

            # save best model
            if mean_val_loss < self.best_val_loss:
                self.best_val_loss = mean_val_loss
                self.model.save(self.model.get_model_path(self.log_dir))

        # at final epoch, print performance on test set
        if self.test_datasets:
            self.print_meta_info(test_meta, mode="test")

        # average losses
        epoch_mean_loss = epoch_mean_loss / self.cfg.epochs
        epoch_mean_val_loss = epoch_mean_val_loss / self.cfg.epochs

        return epoch_mean_loss, epoch_mean_val_loss

    def evaluate(self, dataloader: DataLoader | None = None):
        """
        Evaluate the model
        """
        mean_eval_loss = 0
        meta_eval = {}

        if dataloader is None:
            dataloader = self.validation_dataloader
        batch_number = len(dataloader)

        for batch_idx, inputs in enumerate(dataloader):
            loss, meta = self.model.evaluate(model_in=inputs[:5], target=inputs[5], eval_in=inputs[6:])
            mean_eval_loss += loss

            if batch_idx == 0:
                meta_eval = meta
            else:
                for key, value in meta.items():
                    try:
                        meta_eval[key] += value
                    except KeyError:
                        meta_eval[key] = value

        # average meta_eval
        for key, value in meta_eval.items():
            meta_eval[key] = value / batch_number
        mean_eval_loss = mean_eval_loss / batch_number
        # print meta information
        self.print_meta_info(meta_eval)

        return mean_eval_loss, meta_eval

    def predict(self, model_in) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the trajectory
        """
        with torch.no_grad():
            model_out = self.model.forward(model_in)
        return model_out

    def print_meta_info(self, meta: dict, mode: str = "eval"):
        # Print meta information
        table = PrettyTable()
        table.field_names = ["Loss", "Value"]
        table.align["Loss"] = "l"
        table.align["Value"] = "r"
        for key, value in meta.items():
            table.add_row((key, f"{value:.4f}"))
        print(f"[INFO] {mode} Results\n", table)

    """
    Private functions
    """

    def _init_looging_dir(self):
        # specify directory for logging experiments
        self.log_root_path = os.path.join("logs", "fdm", self.cfg.experiment_name)
        self.log_root_path = os.path.abspath(self.log_root_path)
        print(f"[INFO] Logging experiment in directory: {self.log_root_path}")

    def _init_logging(self):
        # specify directory for logging runs
        log_name = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        if self.cfg.run_name:
            log_name += f"_{self.cfg.run_name}"
        self.log_dir = os.path.join(self.log_root_path, log_name)

        # dump the configuration into log-directory
        dump_yaml(os.path.join(self.log_dir, "params", "trainer_cfg.yaml"), self.cfg)

        # init wandb logging
        if self.cfg.logging:
            os.environ["WANDB_API_KEY"] = self.cfg.wb_api_key
            os.environ["WANDB_MODE"] = "online"

            try:
                wandb.init(
                    project=self.cfg.experiment_name,
                    entity=self.cfg.wb_entity,
                    name=log_name,
                    config=self.cfg.to_dict(),
                    dir=self.log_dir,
                    mode=self.cfg.wb_mode,
                )
                wandb.watch(self.model)
            except:  # noqa: E722
                print("[WARNING: Wandb not available")
