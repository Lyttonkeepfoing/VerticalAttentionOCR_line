from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
from torchvision.transforms import RandomPerspective
from basic.transforms import SignFlipping, DPIAdjusting, Dilation, Erosion, ElasticDistortion, RandomTransform
from basic.utils import LM_str_to_ind
import os
import numpy as np
import pickle
from PIL import Image
import cv2
import copy
import torch