import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize, rgb2hex
from matplotlib.colorbar import ColorbarBase

# # Generate cost values ranging from -150 to 0
cost_values = np.array([
    [ -48.2131],  # red
    [ -70.3419],  # orange
    [-105.3915],  # yellow
    [ -24.8634],  # blue
    [ -33.0853],  # purple
    [ -71.8920],  # gold
    [ -44.2873],  # grey
    [ -66.5247],  # dark gold
    [-150.3916],  # pink
    [-10.12564],  # green
])

# Create a Normalize object for the colormap
norm = Normalize(vmin=cost_values.min(), vmax=cost_values.max())

# Use the RdYlGn colormap
colormap = plt.cm.RdYlBu

# Map cost values to colors
colors = colormap(norm(cost_values))

# Print the colors in hex
hex_colors = [rgb2hex(color) for color in colors]
print("Hex colors:", hex_colors)

# Plot the colorbar
fig, ax = plt.subplots(figsize=(8, 1))
fig.subplots_adjust(bottom=0.5)

# Create a colorbar
colorbar = ColorbarBase(ax, cmap=colormap, norm=norm, orientation='horizontal')

# Remove axis ticks and labels
ax.axis('off')

# Save the colorbar to a file
plt.savefig('colorbar.png', dpi=300, bbox_inches='tight', pad_inches=0)

# Display the plot
plt.show()


# colorbar for the sim planning plot
colormap = sns.color_palette("RdYlBu", as_cmap=True)

# Plot the colorbar
fig, ax = plt.subplots(figsize=(8, 1))
fig.subplots_adjust(bottom=0.5)

# Create a colorbar
colorbar = ColorbarBase(ax, cmap=colormap, orientation='horizontal')

# Remove axis ticks and labels
ax.axis('off')

# Save the colorbar to a file
plt.savefig('colorbar_sim_plan.png', dpi=300, bbox_inches='tight', pad_inches=0)

# Display the plot
plt.show()