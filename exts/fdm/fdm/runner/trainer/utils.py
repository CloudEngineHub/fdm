# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from torch.utils.data import DataLoader


def combined_dataloader(dataloader1: DataLoader, dataloader2: DataLoader):
    """
    Combine two dataloaders by alternating between them.

    If one dataloader is exhausted before the other, the remaining batches from the longer dataloader
    will continue to be yielded.

    The second returned argument is a boolean to indicate if the first or second dataloader is the data source.
    """
    for batch1, batch2 in itertools.zip_longest(dataloader1, dataloader2, fillvalue=None):
        if batch1 is not None:
            yield batch1, False
        if batch2 is not None:
            yield batch2, True
