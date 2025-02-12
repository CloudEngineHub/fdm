# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = torch.inf
        self.delta = delta

    def __call__(self, val_loss) -> bool:
        """
        Check if the model should stop training.

        Args:
            val_loss (float): The validation loss of the model.
        Returns:
            bool: True if the model should stop training, False otherwise.
        """

        if val_loss > self.best_score + self.delta:
            self.counter += 1
            print(f"[INFO] EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = val_loss
            self.counter = 0

        return False

    def reset(self):
        """Reset the early stopping counter."""
        self.counter = 0
        self.best_score = torch.inf
