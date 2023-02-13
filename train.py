from dataset import SegmentationDataset
from model import UNet
from config import IMAGE_DATASET_PATH, MASK_DATASET_PATH, TEST_SPLIT, \
    TEST_PATHS
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os


################################################################################


def main():
    imagePaths = sorted(list(paths.list_images(IMAGE_DATASET_PATH)))
    maskPaths = sorted(list(paths.list_images(MASK_DATASET_PATH)))
    
    print(len(imagePaths), len(maskPaths))
    return

    split = train_test_split(imagePaths, maskPaths, test_size=TEST_SPLIT, 
        random_state=42)
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]
    
    print("[INFO] saving testing image paths...")
    f = open(TEST_PATHS, "w")
    f.write("\n".join(testImages))
    f.close()


if __name__ == "__main__":
    main()
