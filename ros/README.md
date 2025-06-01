# FDM Planning ROS Node

## Overview

The FDM Planning ROS Node is a ROS node that integrates the FDM planning framework with ROS.
It provides a ROS interface for the FDM planning node and allows for easy integration with other ROS nodes.

The node is developed for ANYmal D being on Release 24.04.

## Installation

For details on the ROS integration, see the [README](../README.md).

## Usage

### Launch Files

The `planner.launch` file launches the FDM planning node and the necessary ROS nodes for the planner.
The `record.launch` file logs all relevant data from the robot for later analysis and model fine-tuning.

### Navigation

Under `Tools` select the `WaypointTool` and mark a pose somewhere in the environment. You should then see the robot
to start navigating and walking towards the goal.
