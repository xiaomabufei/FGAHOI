import torch
import torch.nn.functional as F


verb_labels = torch.tensor([0], dtype=torch.int64)
verb_labels = F.one_hot(verb_labels, num_classes=117)
list = ['a','b']
index = list.index('a')
a = 0