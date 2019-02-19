import torch

from utils import load_config

if __name__ == '__main__':
    config = load_config("configs/config.py")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    checkpoint_path = "mobilenet_ssd/19-02-14_00-07/saved/weights_iter138000.pth"

    model = torch.load(checkpoint_path, map_location="cpu")
    model.to(DEVICE)

    with torch.no_grad():
        model.eval()
        params = {"tensorboard": False}
        config.validator(model, params)
