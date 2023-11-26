import nibabel as nib
import numpy as np
from . import functions as fc


# ------------ LOAD IMAGES
def load_nifti(path_name, transpose=False):
    imag_nifti = nib.load(path_name)
    img_data = imag_nifti.get_fdata()
    if transpose:
        img_data = np.transpose(img_data, (1, 0, 2))
    return img_data, imag_nifti.affine


# ------------ DATA TYPE CONVERSIONS 
def nifti2cv(nifti, layer, color_image=False):  
    return fc.img_float2unit8(nifti[:,:, layer].T, color_image=color_image)

# ------------ SAVE IMAGES
def save_nifit(image_np, affine, img_path_name, transpose=False):
    if transpose:
        image_np = np.transpose(image_np, (1, 0, 2)) 

    img_nifti = nib.Nifti1Image(image_np, affine=affine)
    nib.save(img_nifti, img_path_name)





    


