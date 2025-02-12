# Forward Dynamics Model (FDM)

## Overview

The FDM learns to predict the SE2 state of the robot into the future given the next actions.


## Implementation Details

### Logic of the step function when environment in collision

1. Collision is detected but the environment is not reset yet
2. New actions is sampled
3. Current state in collision is saved in a buffer with the future action that will be applied from there onwards
4. The environment is reset and the action will be applied until the next buffer update
