import os
import time
import argparse

import torch

from utils import AverageMeter, load_config
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Train detector')
parser.add_argument('--config', type=str, help='Path to model config')
args = parser.parse_args()

if __name__ == '__main__':
    config = load_config(args.config)

    clip_norm = config.clip_norm if hasattr(config, 'clip_norm') else None

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    savedir = os.path.join(config.logdir, "saved")
    traindir = os.path.join(config.logdir, "train")
    evaldir = os.path.join(config.logdir, "eval")
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(traindir, exist_ok=True)
    os.makedirs(evaldir, exist_ok=True)
    train_writer = SummaryWriter(traindir)
    eval_writer = SummaryWriter(evaldir)

    model = config.model
    model.to(DEVICE)
    model.train()





    epoch = 0
    iteration = 0
    loss_avg = None
    batch_time = AverageMeter()
    end = time.time()
    while True:
        for i, data in enumerate(config.train_loader):
            config.scheduler.step(iteration)

            images, targets = data
            images = images.to(DEVICE)

            config.optimizer.zero_grad()
            model_out = model(images)
            losses = config.criterion(model_out, targets)
            losses["loss"].backward()
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            config.optimizer.step()

            if loss_avg is None:
                loss_avg = {loss_name: AverageMeter() for loss_name in losses}
            for loss_name, loss_val in losses.items():
                loss_avg[loss_name].update(loss_val.item())
            batch_time.update(time.time() - end)


            iteration += 1
            if iteration > config.max_iterations:
                exit()

            if iteration % config.debug_steps == 0:
                print(f"Epoch: {epoch}, Step: {iteration}, Batch time {batch_time.avg:.3f},", end=" ")
                for loss_name, loss_val in loss_avg.items():
                    print(f"{loss_name}: {loss_val.avg:.4f}", end=" ")
                print()
                for loss_name, loss_val in loss_avg.items():
                    train_writer.add_scalar("losses/" + loss_name, loss_val.avg, iteration)

                for param_group in config.optimizer.param_groups:
                    train_writer.add_scalar("lr", param_group['lr'], iteration)

                for loss_name, loss_val in loss_avg.items():
                    loss_val.reset()
                batch_time.reset()

            if iteration % config.val_steps == 0:
                model.eval()
                with torch.no_grad():
                    params = {
                        "epoch": epoch,
                        "iteration": iteration,
                        "savedir": savedir,
                        "eval_writer": eval_writer,
                        "tensorboard": True
                    }
                    config.validator(config.val_loader, model, params)

                model_path = os.path.join(savedir, f"weights_iter{iteration}.pth")
                torch.save(model, model_path)

                model.train()

            end = time.time()

        epoch += 1