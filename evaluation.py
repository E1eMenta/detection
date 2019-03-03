import torch
import argparse

from utils import load_config

parser = argparse.ArgumentParser(description='Evaluate trained detector')
parser.add_argument('--config', required=True, type=str, help='Path to model config')
parser.add_argument('--checkpoint', required=True, type=str, help='Path to model')
args = parser.parse_args()


if __name__ == '__main__':
    config = load_config(args.config)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = torch.load(args.checkpoint, map_location="cpu")
    model.to(DEVICE)

    with torch.no_grad():
        model.eval()
        params = {"tensorboard": False}
        config.validator(config.val_loader, model, params)
