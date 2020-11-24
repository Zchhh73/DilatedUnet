import os
import h5py
import cv2

import numpy as np
from PIL import Image


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


# ------------Path of the images --------------------------------------------------------------
# train
original_imgs_train = "F:\\Verse_Data\\train_data\\img\\"
groundTruth_imgs_train = "F:\\Verse_Data\\train_data\\mask\\"

# test
# original_imgs_test = "./DRIVE/test/1st_manual/"
# groundTruth_imgs_test = "./DRIVE/test/1st_manual/"

# ---------------------------------------------------------------------------------------------

Nimgs = 25792
channels = 3
height = 512
width = 512
dataset_path = "./dataset"


def get_img_datasets(imgs_dir, train_test="null"):
    imgs = np.empty((Nimgs, height, width, channels))
    for path, subdirs, files in os.walk(original_imgs_train):
        for i in range(len(files)):
            # original
            print("original image: " + files[i])
            img = Image.open(imgs_dir + files[i])
            imgs[i] = np.asarray(img)
            print("==========================")

    print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    # reshaping for my standard tensors
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    assert (imgs.shape == (Nimgs, channels, height, width))
    return imgs

def get_mask_datasets(groundTruth_dir, train_test="null"):
    groundTruth = np.empty((Nimgs, height, width, channels))
    for path, subdirs, files in os.walk(original_imgs_train):
        for i in range(len(files)):
            # corresponding ground truth
            groundTruth_name = files[i]
            print("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
            print("==========================")
    groundTruth = np.reshape(groundTruth, (Nimgs, 1, height, width))
    assert (groundTruth.shape == (Nimgs, 1, height, width))
    return groundTruth


if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
# getting the training datasets
imgs_train = get_img_datasets(original_imgs_train,"train")
# groundTruth_train = get_mask_datasets(groundTruth_imgs_train, "train")
print("saving train datasets")
write_hdf5(imgs_train, dataset_path + "Verse_dataset_img_train.hdf5")
# write_hdf5(groundTruth_train, dataset_path + "Verse_dataset_groundTruth_train.hdf5")

# # getting the testing datasets
# imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test, groundTruth_imgs_test,
#                                                               borderMasks_imgs_test, "test")
# print("saving test datasets")
# write_hdf5(imgs_test, dataset_path + "DRIVE_dataset_imgs_test.hdf5")
# write_hdf5(groundTruth_test, dataset_path + "DRIVE_dataset_groundTruth_test.hdf5")
