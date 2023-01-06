from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import numpy as np
import copy
import torch
import torch.utils.data
import torchvision
import math
import torch.nn.functional as F
import os

with open("/mnt/gluster/ssd/datasets/HAKE/proposed_challenge/dc_hoi_list.json", 'r') as f:
    hoi = json.load(f)

corr = np.zeros((93,74))
for anno in hoi:
    row = anno['verb_id']
    clom = anno['object_index']
    if corr[row][clom]==0:
        corr[row][clom]=1
np.save('/mnt/gluster/ssd/datasets/HAKE/proposed_challenge/corre_dc', corr)
print('done')
