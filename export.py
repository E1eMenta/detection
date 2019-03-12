import argparse

import torch
import torch.nn as nn

from utils import load_config

parser = argparse.ArgumentParser(description='Export trained detector')
parser.add_argument('--config', required=True, type=str, help='Path to model config')
parser.add_argument('--checkpoint', required=True, type=str, help='Path to model')
parser.add_argument('--save', required=True, type=str, help='Path to save converte model')
parser.add_argument('--type', default='onnx', type=str, help="Type of converted model: 'onnx' or 'torch'. "
                                                              "Default: 'onnx'")
args = parser.parse_args()

class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        model_out, _ = self.model(inputs)
        return model_out

if __name__ == '__main__':
    config = load_config(args.config)

    model = torch.load(args.checkpoint, map_location='cpu')
    model.eval()

    inputs, _ = next(iter(config.val_loader))

    if args.type == 'torch':
        model_jit = torch.jit.trace(model, inputs)
        model_jit.save(args.save)
    elif args.type == 'onnx':
        model = ONNXWrapper(model)
        model.eval()
        output_example = model(inputs)
        torch.onnx._export(model, inputs, args.save, export_params=True, verbose=True)
    else:
        raise Exception('Export type "{}" is unsupported'.format(args.type))