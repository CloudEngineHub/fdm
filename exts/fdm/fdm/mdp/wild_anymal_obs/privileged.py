

import torch

from .observation_base import ObservationBase
from .raisim_conversion import isaac_raisim_batched_conversion, isaac_raisim_foot_conversion


class PrivilegedObservation(ObservationBase):
    def __init__(self, num_envs=1, simulation_dt=0.0025, control_dt=0.02, device="cuda"):
        super().__init__(simulation_dt, control_dt)
        self.obs_dim = 50
        self.dt = control_dt
        self.num_envs = num_envs
        self.device = device

        mean_foot_contact_force = torch.tensor([0, 0, 80] * 4)
        mean_contact_normal = torch.tensor([0, 0, 1] * 4)
        self.mean = torch.cat([
            torch.ones([4]) * 0.5,
            mean_foot_contact_force,
            mean_contact_normal,
            torch.ones([4]) * 0.6,
            torch.ones([8]) * 0.5,
            torch.zeros([6]),
            torch.zeros([4]),
            torch.zeros([1]),
        ]).to(device)

        std_foot_contact_force = torch.tensor([0.01, 0.01, 0.02] * 4)
        std_contact_normal = torch.tensor([5.0, 5.0, 20.0] * 4)
        self.std_inv = torch.cat([
            torch.ones([4]) * 3,
            std_foot_contact_force,
            std_contact_normal,
            torch.ones([4]) * 2,
            torch.ones([8]) * 2,
            torch.ones([6]) * 0.1,
            torch.ones([4]) * 3.0,
            torch.ones([1]) * 5.0,
        ]).to(device)
        self.obs = torch.tile(self.mean, (num_envs, 1)).to(device)
        self.std = 1 / self.std_inv
        self.air_time = torch.zeros(self.num_envs, 4).to(self.device)
        self.stance_time = torch.zeros(self.num_envs, 4).to(self.device)

    def update_obs(
        self,
        foot_contact_forces,
        foot_contact_normals,
        foot_frictions,
        shank_contact_states,
        thigh_contact_states,
        body_external_force,
        body_external_torque,
        additional_mass,
    ):

        thigh_shank_contact = torch.cat([thigh_contact_states.T, shank_contact_states.T], dim=0).reshape(
            self.num_envs, -1
        )
        foot_contact_states = foot_contact_forces[:, :, 2] > 1.0
        foot_contact_forces = foot_contact_forces.clip(
            min=torch.tensor([-50, -50, -20]).to(self.device), max=torch.tensor([50, 50, 180]).to(self.device)
        )
        # contact_filt = torch.logical_or(foot_contact_states, self.last_contacts)
        # self.last_contacts = contact
        air_time_obs = self._update_air_time(foot_contact_states)
        self.obs = torch.cat(
            [
                foot_contact_states.reshape(self.num_envs, -1),  # 4
                foot_contact_forces.reshape(self.num_envs, -1),  # 12
                foot_contact_normals.reshape(self.num_envs, -1),  # 12
                foot_frictions.reshape(self.num_envs, -1),  # 4
                thigh_shank_contact.reshape(self.num_envs, -1),  # 8
                body_external_force.reshape(self.num_envs, -1),  # 3
                body_external_torque.reshape(self.num_envs, -1),  # 3
                air_time_obs.clip(-3.0, 3.0).reshape(self.num_envs, -1),  # 4
                additional_mass.reshape(self.num_envs, -1),  # 1
            ],
            dim=1,
        )
        if self.is_training:
            self.add_noise_to_obs()

    # Overwrite
    def get_obs(self, *args, use_raisim_order=False):
        obs = (self.obs - self.mean) * self.std_inv
        if use_raisim_order:
            obs = isaac_raisim_batched_conversion(obs, start_indices=[4, 16], batch_size=3)
            obs = isaac_raisim_foot_conversion(obs, start_indices=[0, 28, 46])
            obs = isaac_raisim_batched_conversion(obs, start_indices=[32, 40], batch_size=2)
        return obs

    def reset(self, env_idx):
        self.obs[env_idx] = self.obs_mean

    def _update_air_time(self, foot_contact_states):
        self.air_time += self.dt
        self.air_time *= ~foot_contact_states
        self.stance_time += self.dt
        self.stance_time *= foot_contact_states
        air_time_obs = self.air_time - self.stance_time
        return air_time_obs

    def add_noise_to_obs(self):
        return None


if __name__ == "__main__":
    pass
