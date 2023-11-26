
import cv2 as cv
import numpy as np
import pathlib
import os
import functions.functions as fc
import functions.nifti_functions as nfc

# # ------------ IMAGES
FOLDER_IMAGES_OASIS2 = "data/OAS2_RAW_PART2"
FOLDER_PATIENT = "OAS2_{0:04}_MR{1}"
FOLDER_RAW = "RAW"
FOLDER_REGISTERED= "registered"

NAME_ORIGINAL_VOLUME = "mpr-1.nifti.hdr"
NAME_NSK_VOLUME = "nsk_mpr-1.nii"
NAME_HISTMATCHED = 'matched_mpr-1.nii'


NAME_BRAIN_MASK = "mask_mpr-1.nifti.nii"
NAME_SPM_PROB_LABEL = "c{0}mpr-1.nifti.nii"

NAME_PREDICTED_FUP = "predicted{0}_mpr-1.nii"

IMG_NUMBERS_TRAIN = [100, 101, 102, 103, 104, 105, 106, 108, 109, 111, 112, 113, 114, 116, 117]
IMG_NUMBERS_TEST = [9999, 118, 119, 120]

TISSUE_CSF = "csf"
TISSUE_GM = "gm"
TISSUE_WM = "wm"
TISSUE_ALL = "all"

MODELS_FOLDER = "models"

notebook = False

def get_base_folder(absolute=True):
    if notebook:
        return "/content/drive/MyDrive/MAIA/S3/Computer Aided Diagnosis/lecture_activity/code/"
    else:
        if absolute:
            return str(pathlib.Path(__file__).parent.resolve()).replace(os.sep, "/") + "/"
        else:
            return ""


# ---- paths
def get_patient_path(p_index, longitudinal=1, sub_folder=FOLDER_REGISTERED):
    patient_path = os.path.join(get_base_folder(), FOLDER_IMAGES_OASIS2, FOLDER_PATIENT.format(p_index, longitudinal), sub_folder)
    return patient_path

def get_model_path_name(name):
    model_path_name = os.path.join(get_base_folder(), MODELS_FOLDER, name)
    return model_path_name

# --- loaders
def load_original_volume(p_index, longitudinal=1, transpose=False):
    img_path_name = os.path.join(get_patient_path(p_index, longitudinal=longitudinal, sub_folder=FOLDER_RAW), NAME_ORIGINAL_VOLUME)
    img, affine = nfc.load_nifti(img_path_name, transpose=transpose)
    return img[...,0], affine

def load_nsk_volume(p_index, longitudinal=1, sub_folder=FOLDER_REGISTERED, transpose=False):
    img_path_name = os.path.join(get_patient_path(p_index, longitudinal=longitudinal, sub_folder=sub_folder), NAME_NSK_VOLUME)
    img, affine = nfc.load_nifti(img_path_name, transpose=transpose)
    return img, affine

def load_predicted_volume(p_index, longitudinal=1, sub_folder=FOLDER_REGISTERED, tissue=0, transpose=False):
    img_path_name = os.path.join(get_patient_path(p_index, longitudinal=longitudinal, sub_folder=sub_folder), NAME_PREDICTED_FUP.format(tissue))
    img, affine = nfc.load_nifti(img_path_name, transpose=transpose)
    return img, affine


def load_brain_mask(p_index, longitudinal=1, sub_folder=FOLDER_REGISTERED, transpose=False):
    img_path_name = os.path.join(get_patient_path(p_index, longitudinal=longitudinal, sub_folder=sub_folder), NAME_BRAIN_MASK)
    img, affine = nfc.load_nifti(img_path_name, transpose=transpose)
    return img, affine

def load_spm_labels(p_index, longitudinal=1, sub_folder=FOLDER_REGISTERED, tissue=1, transpose=False):
    if tissue <= 0:
        img_gm_path_name = os.path.join(get_patient_path(p_index, longitudinal=longitudinal, sub_folder=sub_folder), NAME_SPM_PROB_LABEL.format(1))
        img_wm_path_name = os.path.join(get_patient_path(p_index, longitudinal=longitudinal, sub_folder=sub_folder), NAME_SPM_PROB_LABEL.format(2))
        img_csf_path_name = os.path.join(get_patient_path(p_index, longitudinal=longitudinal, sub_folder=sub_folder), NAME_SPM_PROB_LABEL.format(3))
        
        img_prob_label_gm, _ = nfc.load_nifti(img_gm_path_name, transpose=transpose)
        img_prob_label_wm, _ = nfc.load_nifti(img_wm_path_name, transpose=transpose)
        img_prob_label_csf, _ = nfc.load_nifti(img_csf_path_name, transpose=transpose)

        return img_prob_label_gm, img_prob_label_wm, img_prob_label_csf
    else:
        img_path_name = os.path.join(get_patient_path(p_index, longitudinal=longitudinal, sub_folder=sub_folder), NAME_SPM_PROB_LABEL.format(tissue))
        img_label, affine = nfc.load_nifti(img_path_name, transpose=transpose)
        return img_label, affine


# savers
def save_volume(img, affine, img_name, p_index, longitudinal=1, sub_folder=FOLDER_REGISTERED, transpose=False):
    img_path_name = os.path.join(get_patient_path(p_index, longitudinal=longitudinal, sub_folder=sub_folder), img_name)
    nfc.save_nifit(img, affine, img_path_name, transpose=transpose)
