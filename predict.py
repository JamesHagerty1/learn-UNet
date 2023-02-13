from config import PRED_PATH, MASK_DATASET_PATH, INPUT_IMAGE_HEIGHT, DEVICE, \
    THRESHOLD, TEST_PATHS, MODEL_PATH
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os


# TBD, PLOT MULTIPLE ON SAME PLOT -- BUT EVERYTHING WORKS FINE


def prepare_plot(origImage, origMask, predMask):
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	figure.tight_layout()
	plt.savefig(PRED_PATH) # figure.show()
	

def make_predictions(model, imagePath):
	model.eval()
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it to float 
        # data type, and scale its pixel values
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0
		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (128, 128))
		orig = image.copy()
		
		# find the filename and generate the path to ground truth mask
		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(MASK_DATASET_PATH, filename)
		# load the ground-truth segmentation mask in grayscale mode and resize 
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_HEIGHT))
		
		# make the channel axis to be the leading one, add a batch dimension, 
        # create a PyTorch tensor, and flash it to the current device
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(DEVICE)
	
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()
		predMask = (predMask > THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)
		
		prepare_plot(orig, gtMask, predMask)
		

def main():
    imagePaths = open(TEST_PATHS).read().strip().split("\n")
    imagePaths = np.random.choice(imagePaths, size=10)
    unet = torch.load(MODEL_PATH).to(DEVICE)
    for path in imagePaths:
        # make predictions and visualize the results
        make_predictions(unet, path)

if __name__ == "__main__":
	main()
	