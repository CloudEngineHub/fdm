

import torch
from abc import abstractclassmethod


class ObservationBase:
    def __init__(self, simulation_dt=0.0025, control_dt=0.02, is_training=False):
        self.simulation_dt = simulation_dt
        self.control_dt = control_dt
        self.is_training = is_training

    @abstractclassmethod
    def substep_update(self, **kwargs):
        """Update State for each substep"""
        raise NotImplementedError

    @abstractclassmethod
    def update(self, **kwargs):
        """Update observation"""
        raise NotImplementedError

    def get_obs(self) -> torch.Tensor:
        return (self.obs - self.mean) / self.std

    @property
    def num_obs(self) -> int:
        return self.obs.shape[1]
