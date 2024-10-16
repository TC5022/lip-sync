import numpy as np
import imageio

# Create a simple image array
image = np.zeros((10, 10), dtype=np.uint8)

# Save the image as a GIF
imageio.mimsave('test.gif', image)
