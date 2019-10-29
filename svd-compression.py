from matplotlib.image import imread
import math
import matplotlib.pyplot as plt
import numpy as np


'''
Input
'''

# Image filename
img_filename = "lena.bmp"

# Set compression parameter in [0, 1] ('0' stands for no compression)
COMP_PARAM = 0.5


'''
Processing
'''

# Read RGB image into array
img = imread(img_filename)

# Change image data-type to 'float'
img = img.astype('float64')

# Get image resolution
resolution = img.shape

# Get the smallest dimension
m = min(resolution[0:2])

# Maximum feasible rank to achieve compression
max_rank = math.floor(m ** 2 / (1 + 2 * m))

# Number of singular values to keep, for given compression parameter
r = max(math.ceil(max_rank * (1 - COMP_PARAM)), 1)

# Initialize SVD matrices
U = np.full_like(img, 0)
S = np.full_like(img, 0)
VT = np.full_like(img, 0)

# Compute SVD channel-wise
for i in range(3):
    u, s, vt = np.linalg.svd(img[:, :, i], full_matrices=True)
    U[:, :, i] = u
    S[0:m, 0:m, i] = np.diag(s)
    VT[:, :, i] = vt

# Initialize compressed image
img_comp = np.full_like(img, 0, dtype='float64')

# Reconstruct compressed image channel-wise
for j in range(3):
    img_comp[:, :, j] = U[:, 0:r, j] @ S[0:r, 0:r, j] @ VT[0:r, :, j]


'''
Results
'''

# Calculate compression rate
img_bytes = 3 * m ** 2
img_bytes_comp = 3 * (2 * m * r + r)
comp_rate = img_bytes_comp / img_bytes


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(255 / math.sqrt(mse))


# Calculate PSNR
img_psnr = psnr(img, img_comp)

# Print results
results = "Rate: %0.2f%%\nPSNR:%0.2f dB" % (100 * (1 - comp_rate), img_psnr)
print(results)

# Plot images
fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
plt.imshow(img.astype(np.uint8))
a.set_title("Original: %d kB" % round(img_bytes / 1024))
plt.xticks([], [])
plt.yticks([], [])
a = fig.add_subplot(1, 2, 2)
plt.imshow(img_comp.astype(np.uint8))
a.set_title("Compressed: %d kB" % round(img_bytes_comp / 1024))
plt.figtext(.5, .05, results, horizontalalignment='center', fontsize=14)
plt.xticks([], [])
plt.yticks([], [])
plt.show()
