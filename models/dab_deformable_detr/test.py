import torch

a = torch.rand([2,1])
b = torch.rand([2,1])
c = torch.max(a,b)


print(a.shape)