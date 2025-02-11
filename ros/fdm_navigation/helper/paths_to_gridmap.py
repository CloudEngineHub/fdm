# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import string
import torch

import cupy as cp


def path_min_cost_kernel(width, height, resolution):
    path_min_cost_kernel = cp.ElementwiseKernel(
        in_params="raw U seg_start, raw U seg_stop, raw U seg_cost, raw U nr_seg",
        out_params="raw U gridmap_out",
        preamble=string.Template("""
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${height} * ${width};
                return layer * layer_n + idx;
            }
            __device__ bool is_inside_map(int x, int y) {
                return (x >= 0 && y >= 0 && x<${height} && y<${width});
            }
            __device__ float get_l2_distance(int x0, int y0, int x1, int y1) {
                float dx = x0-x1;
                float dy = y0-y1;
                return sqrt( dx*dx + dy*dy);
            }

            __device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
                float old;
                old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
                    __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
                return old;
            }

            """).substitute(height=height, width=width),
        operation=string.Template("""

            int x0 = seg_start[i*2];
            int y0 = seg_start[i*2 + 1 ];

            int x1 = seg_stop[i*2];
            int y1 = seg_stop[i*2 + 1 ];

            // bresenham algorithm to iterate over cells in line between camera center and current gridmap cell
            // https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
            int dx = abs(x1-x0);
            int sx = x0 < x1 ? 1 : -1;
            int dy = -abs(y1 - y0);
            int sy = y0 < y1 ? 1 : -1;
            int error = dx + dy;

            // iterate over all cells along line
            while (1){
                // assumption we do not need to check the height for camera center cell
                // if (x0 >= 0 && y0 >= 0 && x0<${height} && y0<${width}){
                atomicMinFloat(&gridmap_out[x0 * ${height} + y0], seg_cost[i]);
                // }

                if (x0 == x1 && y0 == y1){
                    break;
                }

                // computation of next gridcell index in line
                int e2 = 2 * error;
                if (e2 >= dy){
                    if(x0 == x1){
                        break;
                    }
                    error = error + dy;
                    x0 = x0 + sx;
                }
                if (e2 <= dx){
                    if (y0 == y1){
                        break;
                    }
                    error = error + dx;
                    y0 = y0 + sy;
                }
            }

            """).substitute(height=height, width=width, resolution=resolution),
        name="path_min_cost_kernel",
    )
    return path_min_cost_kernel


def path_count_kernel(width, height, resolution):
    path_count_kernel = cp.ElementwiseKernel(
        in_params="raw U seg_start, raw U seg_stop, raw U seg_cost, raw U nr_seg",
        out_params="raw U gridmap_out",
        preamble=string.Template("""
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            __device__ bool is_inside_map(int x, int y) {
                return (x >= 0 && y >= 0 && x<${width} && y<${height});
            }
            __device__ float get_l2_distance(int x0, int y0, int x1, int y1) {
                float dx = x0-x1;
                float dy = y0-y1;
                return sqrt( dx*dx + dy*dy);
            }
            """).substitute(height=height, width=width),
        operation=string.Template("""
            int x0 = seg_start[i*2];
            int y0 = seg_start[i*2 + 1 ];

            int x1 = seg_stop[i*2];
            int y1 = seg_stop[i*2 + 1 ];

            // bresenham algorithm to iterate over cells in line between camera center and current gridmap cell
            // https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
            int dx = abs(x1-x0);
            int sx = x0 < x1 ? 1 : -1;
            int dy = -abs(y1 - y0);
            int sy = y0 < y1 ? 1 : -1;
            int error = dx + dy;

            // iterate over all cells along line
            while (1){
                // assumption we do not need to check the height for camera center cell
                if (x0 >= 0 && y0 >= 0 && x0<${width} && y0<${height}){
                    gridmap_out[x0 * ${height} + y0] += 1;
                }

                if (x0 == x1 && y0 == y1){
                    break;
                }

                // computation of next gridcell index in line
                int e2 = 2 * error;
                if (e2 >= dy){
                    if(x0 == x1){
                        break;
                    }
                    error = error + dy;
                    x0 = x0 + sx;
                }
                if (e2 <= dx){
                    if (y0 == y1){
                        break;
                    }
                    error = error + dx;
                    y0 = y0 + sy;
                }
            }

            """).substitute(height=height, width=width, resolution=resolution),
        name="path_count_kernel",
    )
    return path_count_kernel


class PathsToGridmap:
    def __init__(self, gridmap_size: tuple, resolution: float, device, kernel: str) -> None:
        """
        gridmap_size (tuple): The size of the gridmap in cells (width, height).
        resolution (float): The resolution of the gridmap in meter per cell.
        device (str): The device to run the kernel on.
        kernel (str): The kernel to use for the gridmap. Either "count" or "min_cost".
        """

        self.device = device
        self.gridmap = torch.zeros(gridmap_size, device=device, dtype=torch.float32)
        self.gridmap_shape = torch.tensor(self.gridmap.shape, device=device, dtype=torch.int32)
        self.gridmap = cp.asarray(self.gridmap)
        self.resolution = resolution

        if kernel == "count":
            self.kernel = path_count_kernel(gridmap_size[0], gridmap_size[1], resolution)
            self.reset_gridmap = self.reset_zero

        elif kernel == "min_cost":
            self.kernel = path_min_cost_kernel(gridmap_size[0], gridmap_size[1], resolution)
            self.reset_gridmap = self.reset_max

    def reset_zero(self):
        self.gridmap[:, :] = 0

    def reset_max(self):
        self.gridmap[:, :] = cp.inf

    def __call__(self, paths: torch.Tensor, path_costs: torch.Tensor, gridmap_center: torch.Tensor) -> torch.Tensor:
        """
        Gridmap Convention - origin is at center of gridmap:
        .---> y
        |
        x

        paths (torch.Tensor, shape:=(NR, PATH_LENGTH, 2), dtype=torch.float32): The paths to be converted to gridmap (x,y) coordinates in meter.
        path_costs (torch.Tensor, shape:=(NR), dtype=torch.float32): The costs of the paths.
        gridmap_center (torch.Tensor, shape:=(2), dtype=torch.float32): The center of the gridmap in meter (x,y).
        """

        NR, LENGTH, DIM = paths.shape

        paths[:, :, :] += gridmap_center[None, None, :]
        # paths //= self.resolution  # floor division
        paths = torch.round(paths / self.resolution)

        # clip the paths to max values
        paths[..., 0] = paths[..., 0].clip(0, self.gridmap_shape[0].item() - 1)
        paths[..., 1] = paths[..., 1].clip(0, self.gridmap_shape[1].item() - 1)

        seg_start = paths[:, :-1, :]
        seg_stop = paths[:, 1:, :]

        seg_start = seg_start.reshape(-1, DIM)
        seg_stop = seg_stop.reshape(-1, DIM)

        seg_costs = path_costs[:, None].repeat(1, LENGTH - 1).reshape(-1)

        self.reset_gridmap()

        nr_segs = seg_start.shape[0]

        self.kernel(
            cp.asarray(seg_start),
            cp.asarray(seg_stop),
            cp.asarray(seg_costs),
            int(nr_segs),
            self.gridmap,
            size=(seg_start.shape[0]),
        )
        return torch.from_dlpack(self.gridmap)  # torch.from_numpy(cp.asnumpy(self.gridmap)).to(self.device) #  #
