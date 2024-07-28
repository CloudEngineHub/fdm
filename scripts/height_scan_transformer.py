import matplotlib
import torch

# Set the backend to Qt5Agg (or another suitable backend)
matplotlib.use("Qt5Agg")


import math

############################################################################################################

# Example usage
height_scan = torch.rand(100, 100)  # Example height scan tensor
translation = (0.3, 1.0)  # Translation vector (1m forward, 0m sideways)
rotation_angle = torch.tensor(torch.pi / 3)  # Rotation angle (60 degrees)
height_scan_res = 0.1  # Height scan resolution (10cm)
height_scan_robot_center = [height_scan.shape[0] / 2, 0.5 / height_scan_res]
# Define the bounds of the subregion to extract
x_min, x_max = -0.5, 1.0  # height --> x
y_min, y_max = -1.0, 1.0  # width --> y

# get effective translation
# since in robot frame, the y translation is against the height axis x direction, has to be negative
effective_translation_tensor_x = -translation[1] / height_scan_res + height_scan_robot_center[0]
effective_translation_tensor_y = translation[0] / height_scan_res + height_scan_robot_center[1]

# Create a meshgrid of coordinates
idx_tensor_y, idx_tensor_x = torch.meshgrid(
    torch.arange(x_min / height_scan_res, (x_max / height_scan_res) + 1),
    torch.arange(y_min / height_scan_res, (y_max / height_scan_res) + 1),
    indexing="ij",
)
idx_tensor_x = idx_tensor_x.flatten().float()
idx_tensor_y = idx_tensor_y.flatten().float()

# angle definition for the height scan coordinate system is opposite of the tensor system, so negative
c, s = torch.cos(rotation_angle), torch.sin(rotation_angle)
idx_crop_x = c * idx_tensor_x - s * idx_tensor_y + effective_translation_tensor_x
idx_crop_y = s * idx_tensor_x + c * idx_tensor_y + effective_translation_tensor_y


# filter_idx outside the image
filter_idx = (
    (idx_crop_x >= 0) & (idx_crop_x < height_scan.shape[0]) & (idx_crop_y >= 0) & (idx_crop_y < height_scan.shape[1])
)

# make indexes to integer for indexing
idx_crop_x = idx_crop_x[filter_idx].int()
idx_crop_y = idx_crop_y[filter_idx].int()
idx_tensor_x = idx_tensor_x[filter_idx].int()
idx_tensor_y = idx_tensor_y[filter_idx].int()

# see filtered indexes as mask
mask = torch.zeros_like(height_scan, dtype=torch.bool)
mask[idx_crop_x, idx_crop_y] = True
masked_image = torch.where(mask, height_scan, torch.tensor(-1))

# move idx tensors of the new image to 0,0 in upper left corner
idx_tensor_x += torch.abs(torch.min(idx_tensor_x))
idx_tensor_y += torch.abs(torch.min(idx_tensor_y))

# new_image = torch.zeros((math.ceil((x_max - x_min) / height_scan_res + 1), math.ceil((y_max - y_min) / height_scan_res + 1)))
new_image = torch.zeros(
    (math.ceil((y_max - y_min) / height_scan_res + 1), math.ceil((x_max - x_min) / height_scan_res + 1))
)
new_image[idx_tensor_x, idx_tensor_y] = height_scan[idx_crop_x, idx_crop_y]

print(new_image)

import matplotlib.pyplot as plt

# Visualization using matplotlib
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(height_scan.numpy(), cmap="viridis")
axes[0].set_title("Original Height Scan")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")

axes[1].imshow(new_image.numpy(), cmap="viridis")
axes[1].set_title("Cropped Height Scan")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")

axes[2].imshow(masked_image.numpy(), cmap="viridis")
axes[2].set_title("Masked Height Scan")
axes[2].set_xlabel("X")
axes[2].set_ylabel("Y")

plt.tight_layout()
plt.show()

print("done")
