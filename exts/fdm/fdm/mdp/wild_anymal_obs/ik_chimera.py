

import numpy as np
import torch
import torch.nn as nn


class IK(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.pos_base_to_hip_in_base_frame = np.zeros([4, 3])
        self.pos_hip_to_thigh_in_hip_frame = np.zeros([4, 3])
        self.pos_thigh_to_shank_in_thigh_frame = np.zeros([4, 3])
        self.pos_shank_to_foot_in_shank_frame = np.zeros([4, 3])
        self.pos_base_to_haa_center_in_base_frame = np.zeros([4, 3])
        self.hfe_to_foot_y_offset = np.zeros([4])
        self.haa_to_foot_y_offset = np.zeros([4])

        self.pos_base_to_hip_in_base_frame[0, :] = [0.3, 0.104, 0.0][:]
        self.pos_base_to_hip_in_base_frame[2, :] = [0.3, -0.104, 0.0][:]
        self.pos_base_to_hip_in_base_frame[1, :] = [-0.3, 0.104, 0.0][:]
        self.pos_base_to_hip_in_base_frame[3, :] = [-0.3, -0.104, 0.0][:]

        self.pos_hip_to_thigh_in_hip_frame[0, :] = [0.06, 0.08381, 0.0][:]
        self.pos_hip_to_thigh_in_hip_frame[2, :] = [0.06, -0.08381, 0.0][:]
        self.pos_hip_to_thigh_in_hip_frame[1, :] = [-0.06, 0.08381, 0.0][:]
        self.pos_hip_to_thigh_in_hip_frame[3, :] = [-0.06, -0.08381, 0.0][:]

        self.pos_thigh_to_shank_in_thigh_frame[0, :] = [0.0, 0.1003, -0.285][:]
        self.pos_thigh_to_shank_in_thigh_frame[2, :] = [0.0, -0.1003, -0.285][:]
        self.pos_thigh_to_shank_in_thigh_frame[1, :] = [0.0, 0.1003, -0.285][:]
        self.pos_thigh_to_shank_in_thigh_frame[3, :] = [0.0, -0.1003, -0.285][:]

        self.pos_shank_to_foot_in_shank_frame[0, :] = [0.08795, -0.01305, -0.33797][:]
        self.pos_shank_to_foot_in_shank_frame[2, :] = [0.08795, 0.01305, -0.33797][:]
        self.pos_shank_to_foot_in_shank_frame[1, :] = [0.08795, -0.01305, -0.33797][:]
        self.pos_shank_to_foot_in_shank_frame[3, :] = [0.08795, 0.01305, -0.33797][:]

        self.pos_shank_to_foot_in_shank_frame[:, 2] += 0.0225

        for i in range(4):
            self.pos_base_to_haa_center_in_base_frame[i] = self.pos_base_to_hip_in_base_frame[i]
            self.pos_base_to_haa_center_in_base_frame[i][0] += self.pos_hip_to_thigh_in_hip_frame[i, 0]
            self.hfe_to_foot_y_offset[i] = self.pos_thigh_to_shank_in_thigh_frame[i, 1]
            self.hfe_to_foot_y_offset[i] += self.pos_shank_to_foot_in_shank_frame[i, 1]
            self.haa_to_foot_y_offset[i] = self.hfe_to_foot_y_offset[i]
            self.haa_to_foot_y_offset[i] += self.pos_hip_to_thigh_in_hip_frame[i, 1]

        self.a0 = np.sqrt(self.pos_hip_to_thigh_in_hip_frame[0, 1] ** 2 + self.pos_hip_to_thigh_in_hip_frame[0, 2] ** 2)
        self.haa_offset = np.abs(
            np.arctan2(self.pos_hip_to_thigh_in_hip_frame[0, 2], self.pos_hip_to_thigh_in_hip_frame[0, 1])
        )
        self.a1_squared = (
            self.pos_thigh_to_shank_in_thigh_frame[0, 0] ** 2 + self.pos_thigh_to_shank_in_thigh_frame[0, 2] ** 2
        )
        self.a2_squared = (
            self.pos_shank_to_foot_in_shank_frame[0, 0] ** 2 + self.pos_shank_to_foot_in_shank_frame[0, 2] ** 2
        )

        self.min_reach_sp = np.abs(np.sqrt(self.a1_squared) - np.sqrt(self.a2_squared)) + 0.1
        self.max_reach_sp = np.sqrt(self.a1_squared) + np.sqrt(self.a2_squared) - 0.05
        self.min_reach = np.sqrt(self.haa_to_foot_y_offset[0] ** 2 + self.min_reach_sp**2)
        self.max_reach = np.sqrt(self.haa_to_foot_y_offset[0] ** 2 + self.max_reach_sp**2)
        self.kfe_offset = np.abs(
            np.arctan(self.pos_shank_to_foot_in_shank_frame[0, 0] / self.pos_shank_to_foot_in_shank_frame[0, 2])
        )

        self.pos_base_to_hip_in_base_frame = torch.from_numpy(self.pos_base_to_hip_in_base_frame).to(device)
        self.pos_hip_to_thigh_in_hip_frame = torch.from_numpy(self.pos_hip_to_thigh_in_hip_frame).to(device)
        self.pos_thigh_to_shank_in_thigh_frame = torch.from_numpy(self.pos_shank_to_foot_in_shank_frame).to(device)
        self.pos_shank_to_foot_in_shank_frame = torch.from_numpy(self.pos_shank_to_foot_in_shank_frame).to(device)
        self.pos_base_to_haa_center_in_base_frame = torch.from_numpy(self.pos_base_to_haa_center_in_base_frame).to(
            device
        )
        self.hfe_to_foot_y_offset = torch.from_numpy(self.hfe_to_foot_y_offset).to(device)
        self.haa_to_foot_y_offset = torch.from_numpy(self.haa_to_foot_y_offset).to(device)

        self.a1_squared = torch.tensor(self.a1_squared, requires_grad=False).to(device)
        self.a2_squared = torch.tensor(self.a2_squared, requires_grad=False).to(device)


ik = IK()


def solve_ik(pos_base_to_foot_in_base_frame, limb, device="cuda"):
    # limb order: LF, LH, RF, RH
    num_envs = pos_base_to_foot_in_base_frame.shape[0]
    leg_joints = torch.zeros([num_envs, 3], device=device, requires_grad=False)
    pos_haa_to_foot_in_base_frame = (
        pos_base_to_foot_in_base_frame - ik.pos_base_to_haa_center_in_base_frame.index_select(0, limb)
    )

    d = ik.haa_to_foot_y_offset.index_select(0, limb)
    d_squared = d * d

    reach = torch.norm(pos_haa_to_foot_in_base_frame, dim=1)
    clipped_reach = torch.clip(reach, ik.min_reach, ik.max_reach)
    pos_haa_to_foot_in_base_frame = torch.einsum("bi,b->bi", pos_haa_to_foot_in_base_frame, clipped_reach / reach)

    pos_yz_squared = torch.norm(pos_haa_to_foot_in_base_frame[:, -2:], dim=1) ** 2

    modified_pos_haa_to_foot_in_base_frame = pos_haa_to_foot_in_base_frame.clone()
    modified_pos_haa_to_foot_in_base_frame[:, -2:] = torch.einsum(
        "bi,b->bi", pos_haa_to_foot_in_base_frame[:, -2:], (torch.abs(d) + 0.01) / torch.sqrt(pos_yz_squared)
    )
    modified_pos_haa_to_foot_in_base_frame[:, 0] = torch.min(
        pos_haa_to_foot_in_base_frame[:, 0], torch.tensor(ik.max_reach_sp)
    )
    modified_pos_yz_squared = torch.norm(modified_pos_haa_to_foot_in_base_frame[:, -2:], dim=1) ** 2
    pos_haa_to_foot_in_base_frame = torch.where(
        pos_yz_squared < d_squared,
        modified_pos_haa_to_foot_in_base_frame.transpose(0, 1),
        pos_haa_to_foot_in_base_frame.transpose(0, 1),
    ).transpose(0, 1)
    pos_yz_squared = torch.where(pos_yz_squared < d_squared, modified_pos_yz_squared, pos_yz_squared)

    r_squared = pos_yz_squared - d_squared
    r = torch.sqrt(r_squared)
    delta = torch.atan2(pos_haa_to_foot_in_base_frame[:, 1], -pos_haa_to_foot_in_base_frame[:, 2])
    beta = torch.atan2(r, d)
    qHAA = beta + delta - np.pi / 2
    leg_joints[:, 0] = qHAA

    l_squared = r_squared + pos_haa_to_foot_in_base_frame[:, 0] ** 2
    l = torch.sqrt(l_squared)
    phi1 = torch.acos((ik.a1_squared + l_squared - ik.a2_squared) * 0.5 / (torch.sqrt(ik.a1_squared) * l))
    phi2 = torch.acos((ik.a2_squared + l_squared - ik.a1_squared) * 0.5 / (torch.sqrt(ik.a2_squared) * l))

    qKFE = phi1 + phi2 - ik.kfe_offset
    qKFE = torch.where(limb % 2 == 0, -qKFE, qKFE)
    leg_joints[:, 2] = qKFE

    theta_prime = torch.atan2(pos_haa_to_foot_in_base_frame[:, 0], r)

    qHFE = torch.where(limb % 2 != 0, -phi1 - theta_prime, phi1 - theta_prime)
    leg_joints[:, 1] = qHFE
    if torch.any(torch.isnan(leg_joints)):
        print("Error: nan detected in IK.")
    return leg_joints


def get_foot_height(pi):
    t = pi / 3.14159265359
    dh_lift = torch.where(t < 1.0, -2 * t**3 + 3 * t**2, 2 * (t - 1) ** 3 - 3 * (t - 1) ** 2 + 1)
    dh = torch.where(pi > 0.0, dh_lift, torch.zeros_like(dh_lift))
    return dh


if __name__ == "__main__":
    print(
        solve_ik(
            torch.tensor(
                [
                    [0.3978, 0.2020, -0.5500],
                    [-0.4022, 0.2020, -0.5484],
                    [0.3978, -0.1980, -0.5484],
                    [-0.4022, -0.1980, -0.5500],
                ],
                device="cuda",
            ),
            torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda"),
        )
    )
    print(get_foot_height(torch.tensor([[0.1]])))
