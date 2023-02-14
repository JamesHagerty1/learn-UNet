from config import INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch


################################################################################


class UNet(Module):
	def __init__(self, encChannels=(3, 16, 32, 64), decChannels=(64, 32, 16),
		nbClasses=1, retainDim=True, 
		outSize=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)):
		super().__init__()
		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)
		self.head = Conv2d(decChannels[-1], nbClasses, 1)
		self.retainDim = retainDim
		self.outSize = outSize
		
	# x (batch_size, 3, width, height)
	def forward(self, x):
		# encFeatures[i] (batch_size, ...)
		encFeatures = self.encoder(x)


		# pass the encoder features through decoder making sure that
		# their dimensions are suited for concatenation
		decFeatures = self.decoder(encFeatures[::-1][0],
			encFeatures[::-1][1:])
		# pass the decoder features through the regression head to
		# obtain the segmentation mask
		map = self.head(decFeatures)
		# check to see if we are retaining the original output
		# dimensions and if so, then resize the output to match them
		if self.retainDim:
			map = F.interpolate(map, self.outSize)
		# return the segmentation map
		return map


class Encoder(Module):
	def __init__(self, channels=(3, 16, 32, 64)):
		super().__init__()
		# Default [Block(3, 16), Block(16, 32), Block(32, 64)]
		self.encBlocks = ModuleList(
			[Block(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)])
		self.pool = MaxPool2d(2)
		
	# x (batch_size, 3, width, height)
	def forward(self, x):
		print(" * Encoder f()")
		blockOutputs = []
		for block in self.encBlocks:
			x = block(x)
			blockOutputs.append(x)
			x = self.pool(x)
		return blockOutputs


class Decoder(Module):
	def __init__(self, channels=(64, 32, 16)):
		super().__init__()
		self.channels = channels
		self.upconvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		self.dec_blocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
		
	def crop(self, encFeatures, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		return encFeatures

	def forward(self, x, encFeatures):
		print(" * Decoder f()")
		for i in range(len(self.channels) - 1):
			x = self.upconvs[i](x)
			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)
		return x


class Block(Module):
	def __init__(self, inChannels, outChannels, kernel_size=3):
		super().__init__()
		# Conv2d(in_channels, out_channels, kernel_size, ...)
		# conv1 weight (outChannels, inChannels, kernel_size, kernel_size)
		self.conv1 = Conv2d(inChannels, outChannels, kernel_size)
		self.relu = ReLU()
		# conv1 weight (outChannels, outChannels, kernel_size, kernel_size)
		self.conv2 = Conv2d(outChannels, outChannels, kernel_size)
		self.channels = (inChannels, outChannels)
		
	# ORIG width and height here, referenced in other dim comments
	# x (batch_size, inChannels, width, height)
	def forward(self, x):
		# x (batch_size, outChannels, width-=(kernel_size+1), height-=(kernel_size+1))
		x = self.conv1(x)
		x = self.relu(x)
		# x (batch_size, outChannels, width-=(kernel_size+1), height-=(kernel_size+1))
		x = self.conv2(x)
		return x
	