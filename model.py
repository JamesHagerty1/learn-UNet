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
		
	# x (batch_size, channels, height, width)
	def forward(self, x):
		# encFeatures[i] (batch_size, channels, height', width') 
		# ascending values for channels
		encFeatures = self.encoder(x)
		# x from last Block of encoder pipeline, max channels val for dec or enc
		x = encFeatures[-1]
		# x (batch_size, decChannels[-1], height', width')
		x = self.decoder(x, encFeatures[::-1][1:])
		# Start segmentation map (mask) with Conv2d where channels are reduced 
		# to nbClasses==1 and kernel_size is 1 (per pixel convolutions)
		map = self.head(x)
		if self.retainDim:
			# Set map height and width to desired dims
			map = F.interpolate(map, self.outSize)
		return map


# Storing convolution+pooling passes' results
class Encoder(Module):
	def __init__(self, channels=(3, 16, 32, 64)):
		super().__init__()
		# Block(in_channels, out_channels)
		# Default [Block(3, 16), Block(16, 32), Block(32, 64)]
		self.encBlocks = ModuleList(
			[Block(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)])
		self.pool = MaxPool2d(2)
		
	# x (batch_size, channels, width, height)
	def forward(self, x):
		blockOutputs = []
		for block in self.encBlocks:
			# x (batch_size, channels', width', height')
			x = block(x)
			blockOutputs.append(x)
			# x (batch_size, channels', width'//pool_dim, height'//pool_dim)
			x = self.pool(x)
		return blockOutputs


class Decoder(Module):
	def __init__(self, channels=(64, 32, 16)):
		super().__init__()
		self.channels = channels
		# Default [ConvTranspose2d(64, 32, ...), ConvTranspose2d(32, 16, ...)]
		# self.upconvs[i] weight (in_channels, out_channels, 2, 2)
		# self.upconvs[i] bias (out_channels)
		self.upconvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		# Default [Block(64, 32), Block(32, 16)]
		self.dec_blocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])

	# match dims for upconv'd x and its U-mirrored encFeatures
	def crop(self, encFeatures, x):
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		return encFeatures

	# x (batch_size, channels, height, width)
	def forward(self, x, encFeatures):
		# upconv x and concat it to U-mirrored encFeature, then Block conv x'
		for i in range(len(self.channels) - 1):
			# channels decreases to ConvTranspose2d out_channels, height and 
			# width increase sensitive to upconv kernel 
			x = self.upconvs[i](x)
			encFeat = self.crop(encFeatures[i], x)
			# skip layer concat
			# channels doubles due to concat
			x = torch.cat([x, encFeat], dim=1)
			# channels decreases to Block out_channels, minor height and width
			# decrease from Block convolutions
			x = self.dec_blocks[i](x)
		return x


class Block(Module):
	def __init__(self, inChannels, outChannels, kernel_size=3):
		super().__init__()
		# conv1 weight (outChannels, inChannels, kernel_size, kernel_size)
		# conv1 bias (outChannels)
		self.conv1 = Conv2d(inChannels, outChannels, kernel_size)
		self.relu = ReLU()
		# conv2 weight (outChannels, outChannels, kernel_size, kernel_size)
		# conv2 bias (outChannels)
		self.conv2 = Conv2d(outChannels, outChannels, kernel_size)
		
	# x (batch_size, inChannels, width, height)
	def forward(self, x):
		# Kernel/convolving reduces width and height slightly
		# x (batch_size, outChannels, width', height')
		x = self.conv1(x)
		x = self.relu(x)
		# x (batch_size, outChannels, width', height')
		x = self.conv2(x)
		return x
	