import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import skimage.transform

import os
import sys

import itertools as it
import random
from collections import deque
from time import sleep, time
from tqdm import trange

import vizdoom as vzd



#Down Sample Game Images
#===================================================================================================
def preprocess(img, resolution=(30, 45)):
    """Down-samples game observation image to given resolution"""
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img