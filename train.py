from dataset import SegmentationDataset
from model import UNet
from config import IMAGE_DATASET_PATH, MASK_DATASET_PATH, TEST_SPLIT, \
    TEST_PATHS, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, BATCH_SIZE, PIN_MEMORY, \
    DEVICE, INIT_LR, NUM_EPOCHS
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
    # 4000 data points (organize paths to them)
    imagePaths = sorted(list(paths.list_images(IMAGE_DATASET_PATH)))
    maskPaths = sorted(list(paths.list_images(MASK_DATASET_PATH)))
    split = train_test_split(imagePaths, maskPaths, test_size=TEST_SPLIT, 
        random_state=42)
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]
    
    # reference for inference later
    print("[INFO] saving testing image paths...")
    f = open(TEST_PATHS, "w")
    f.write("\n".join(testImages))
    f.close()

    # torch datasets + dataloaders
    transforms = transforms.Compose([transforms.ToPILImage(),
 	    transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
	    transforms.ToTensor()])
    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
        transforms=transforms)
    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
        transforms=transforms)
    trainLoader = DataLoader(trainDS, shuffle=True,
        batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
        num_workers=os.cpu_count())
    testLoader = DataLoader(testDS, shuffle=False,
        batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
        num_workers=os.cpu_count())

    # function
    unet = UNet().to(DEVICE)
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=INIT_LR)

    # training refs
    trainSteps = len(trainDS) // BATCH_SIZE
    testSteps = len(testDS) // BATCH_SIZE
    H = {"train_loss": [], "test_loss": []}

    # 
    startTime = time.time()
    for e in tqdm(range(NUM_EPOCHS)):
        unet.train()
        
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0
        
        for (i, (x, y)) in enumerate(trainLoader):
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            # perform a forward pass and calculate the training loss
            pred = unet(x)
            loss = lossFunc(pred, y)
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far
            totalTrainLoss += loss
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            unet.eval()
            # loop over the validation set
            for (x, y) in testLoader:
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                # make the predictions and calculate the validation loss
                pred = unet(x)
                totalTestLoss += lossFunc(pred, y)
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

if __name__ == "__main__":
    main()
