import os
import random
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir, basename
from glob import glob
from tqdm import tqdm, trange
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

from .projects.LiteMedSAM.segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from .projects.LiteMedSAM.tiny_vit_sam import TinyViT
import cv2
import torch.nn.functional as F

from matplotlib import pyplot as plt
import argparse
from framework import BaseDetector


class SampleModel(BaseDetector):
    def __init__(self,
                ):
        
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(torch.zeros(1) + 0.0001, dtype=torch.float32), requires_grad=True)
        
    def forward(self, data):
        
        return {'pred': data['input'] * self.weight}
