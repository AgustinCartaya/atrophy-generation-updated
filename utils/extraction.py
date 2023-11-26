import numpy as np
from skimage.util import view_as_windows

def extract_patches(volume, patch_shape, extraction_step):
    patches = view_as_windows(volume, patch_shape, step=extraction_step)
    ndim = len(volume.shape)
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches, ) + patch_shape)
