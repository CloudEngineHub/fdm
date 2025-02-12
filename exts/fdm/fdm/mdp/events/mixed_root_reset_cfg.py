# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.managers import EventTermCfg
from omni.isaac.lab.utils import configclass

from .mixed_root_reset import MixedRootResetEvent


@configclass
class MixedRootResetEventCfg(EventTermCfg):
    """Mix of different reset commands."""

    class_type: type = MixedRootResetEvent

    @configclass
    class SubsetEventTermCfg:
        """Configuration for a subset of the environments."""

        event_term_cfg: EventTermCfg = MISSING
        """The configuration for the command term."""

        ratio: float = MISSING
        """The ratio of environments that this command term should be applied to."""

    terms: dict[str, SubsetEventTermCfg] = MISSING
    """The command terms that should be applied to the environments."""
