import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

SOURCE_DIR = "/scratchdata/processed/desk"

INDEX = 4

rgb = Image.open(f"{SOURCE_DIR}/rgb/{INDEX}.png")
depth = Image.open(f"{SOURCE_DIR}/depth/{INDEX}.png")

rgb = np.array(rgb)
depth = np.array(depth)

plt.imsave("rgb.png", rgb)
plt.imsave("depth.png", depth)