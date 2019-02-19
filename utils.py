import os
import sys
import os.path as osp
from importlib import import_module

import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_config(filename):
    filename = osp.abspath(osp.expanduser(filename))
    if not os.path.exists(filename):
        raise IOError("File {} doesn't exists")
    if filename.endswith('.py'):
        module_name = osp.basename(filename)[:-3]
        if '.' in module_name:
            raise ValueError('Dots are not allowed in config file path.')
        config_dir = osp.dirname(filename)
        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)

        return mod
    else:
        raise IOError('Only py type is supported now!')

def weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if not m.bias is None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            if not m.bias is None:
                m.bias.data.zero_()