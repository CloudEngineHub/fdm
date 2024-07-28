

import torch

# LEGGED GYM
isaac_gym_raisim_indices_joint = torch.cat(
    (torch.arange(0, 3), torch.arange(6, 9), torch.arange(3, 6), torch.arange(9, 12))
)
isaac_raisim_indices_foot = torch.tensor([0, 2, 1, 3])


# isaaclab
# raisim_joint_names_list = ['LF_HAA', 'LF_HFE', 'LF_KFE', 'RF_HAA', 'RF_HFE', 'RF_KFE', 'LH_HAA', 'LH_HFE', 'LH_KFE', 'RH_HAA', 'RH_HFE', 'RH_KFE']
# ISAAC_GYM_JOINT_NAMES = ['LF_HAA', 'LF_HFE', 'LF_KFE', 'LH_HAA', 'LH_HFE', 'LH_KFE', 'RF_HAA', 'RF_HFE', 'RF_KFE', 'RH_HAA', 'RH_HFE', 'RH_KFE']
# isaaclab_joint_names_list = ['LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA', 'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE', 'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']
# raisim_to_isaac_indices_joint = torch.tensor([raisim_joint_names_list.index(curr_joint_name) for curr_joint_name in isaaclab_joint_names_list])
# isaac_to_raisim_indices_joint = torch.tensor([isaaclab_joint_names_list.index(curr_joint_name) for curr_joint_name in raisim_joint_names_list ])

# isaac_raisim_indices_foot = torch.tensor([0, 2, 1, 3])


def convert(x, joint_indices, start_indices=[0]):
    indices = torch.arange(x.shape[1]).to(x.device)
    n_joint = joint_indices.shape[0]
    for idx in start_indices:
        indices[idx : idx + n_joint] = joint_indices.to(x.device) + idx
    return torch.index_select(x, 1, indices)


def isaac_to_raisim_joint_conversion(x, start_indices=[0]):
    indices = isaac_gym_raisim_indices_joint  # isaac_to_raisim_indices_joint
    return convert(x, indices, start_indices=start_indices)


def raisim_to_isaac_joint_conversion(x, start_indices=[0]):
    indices = isaac_gym_raisim_indices_joint  # raisim_to_isaac_indices_joint
    return convert(x, indices, start_indices=start_indices)


def isaac_raisim_foot_conversion(x, start_indices=[0]):
    indices = isaac_raisim_indices_foot
    return convert(x, indices, start_indices=start_indices)


def isaac_raisim_batched_conversion(x, start_indices=[0], batch_size=1):
    # n = x.shape[1] // 4
    n = batch_size
    indices = torch.arange(x.shape[1]).to(x.device)
    batch = torch.cat(
        (torch.arange(0, n), torch.arange(n * 2, n * 3), torch.arange(n, n * 2), torch.arange(n * 3, n * 4))
    )
    for idx in start_indices:
        indices[idx : idx + batch.shape[0]] = batch.to(x.device) + idx
    return torch.index_select(x, 1, indices.to(x.device))


if __name__ == "__main__":
    q = torch.arange(12 * 3).reshape(3, 12)
    print("q ", q)
    rq = isaac_to_raisim_joint_conversion(q)
    print("raisim ", rq)
    gc = torch.zeros(2 * 19).reshape(2, -1)
    gc[:, 7:] = torch.arange(12) + 1
    print("gc ", gc)
    gc_r = isaac_to_raisim_joint_conversion(gc, start_indices=[7])
    print("gc raisim ", gc_r)
    gv = torch.zeros(2 * 18).reshape(2, -1)
    gv[:, 6:] = torch.arange(12) + 1
    gc_gv = torch.cat([gc, gv], dim=1)
    r_gc_gv = isaac_to_raisim_joint_conversion(gc_gv, start_indices=[7, 25])
    i_gc_gv = isaac_to_raisim_joint_conversion(r_gc_gv, start_indices=[7, 25])
    print("gc gv ", gc_gv)
    print("raisim ", r_gc_gv)
    print("isaac ", i_gc_gv)
    phases = torch.arange(4 * 2).reshape(2, 4)
    raisim_phases = isaac_raisim_foot_conversion(phases)
    print("phases ", phases)
    print("raisim phases ", raisim_phases)

    phases = torch.arange(24 * 1).reshape(1, -1)
    raisim_phases = isaac_raisim_foot_conversion(phases, [4, 8, 15])
    print("phases ", phases)
    print("raisim phases ", raisim_phases)

    height_scan = torch.arange(20 * 2).reshape(2, -1)
    print("height scan", height_scan)
    raisim_scan = isaac_raisim_batched_conversion(height_scan, batch_size=height_scan.shape[1] // 4)
    print("raisim scan", raisim_scan)
