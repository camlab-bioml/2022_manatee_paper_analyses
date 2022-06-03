import torch
import math
import torch.nn.functional as F

def true_f(x):
    y = torch.cat([
        F.relu(torch.sin(x * (2 * math.pi))) + torch.randn(x.size()) * 0.3,
        torch.sin((x - 0.05) * (2 * math.pi)) + torch.randn(x.size()) * 0.3,
        torch.sin(x * (2 * math.pi)) + torch.randn(x.size()) * 0.3,
        torch.randn(x.size()) * 0.8 - 2*x,
        2* x.clone().detach() + torch.randn(x.size()) * 0.1,
    ], axis=1)
       
    return y

def get_labels():
    labels = ['Relu sine', 'Shifted sine', 'Sine', 
              'Noise', 'Linear']
    return labels
