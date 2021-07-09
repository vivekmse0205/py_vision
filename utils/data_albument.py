import albumentations
from albumentations import *
from albumentations import Compose
from albumentations.pytorch import ToTensor
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

class TrainAlbumentation():
	def __init__(self):
		self.train_trans = Compose([
		PadIfNeeded(min_height=40, min_width=40, always_apply=True),
		RandomCrop(width=32, height=32,p=1),
		HorizontalFlip(),
		Rotate((-10.0, 10.0)),
		CoarseDropout(max_holes=3,min_holes = 1, max_height=8, max_width=8, p=0.8,fill_value=tuple([x * 255.0 for x in mean]),
                                      min_height=8, min_width=8),
		Normalize(
			mean=[0.485,0.456,0.406],
			std=[0.229,0.224,0.225],
		),
		ToTensorV2()
		])

	def __call__(self, im):
		im = np.array(im)
		im = self.train_trans(image = im)['image']
		return im
 
class TestAlbumentation():
	def __init__(self):
		self.train_trans = Compose([
		Normalize(
			mean=[0.485,0.456,0.406],
			std=[0.229,0.224,0.225],
		),
		ToTensorV2()
		])

	def __call__(self, im):
		im = np.array(im)
		im = self.train_trans(image = im)['image']
		return im
