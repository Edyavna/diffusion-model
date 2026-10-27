"""MNIST diffusion from Lecture_6_full.ipynb: DhariwalUNet + EDM + Euler + CFG.

Run: python diffusion_mnist.py --help
The NVIDIA network implementation and its license are in vendor/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.utils import save_image

from vendor.edm_networks import DhariwalUNet


class EDMPrecond(nn.Module):
    """Turn the raw U-Net output into a clean-image estimate D(x, sigma)."""

    def __init__(self, sigma_data: float, conditional: bool = False):
        super().__init__()
        self.config = dict(sigma_data=float(sigma_data), conditional=conditional)
        self.sigma_data = float(sigma_data)
        self.conditional = conditional
        self.net = DhariwalUNet(
            img_resolution=16, in_channels=1, out_channels=1,
            model_channels=32, channel_mult=[1, 2, 3, 4],
            label_dim=11 if conditional else 0,
        )

    def forward(self, x, sigma, labels=None):
        # [B,1,1,1] makes one noise level broadcast over each image.
        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype).reshape(-1, 1, 1, 1)
        if sigma.numel() not in (1, x.shape[0]) or not torch.all(sigma > 0):
            raise ValueError("sigma must be positive, with one value or one per image")
        sigma = sigma.expand(x.shape[0], 1, 1, 1)
        if self.conditional:
            if labels is None:
                labels = torch.full((x.shape[0],), 10, device=x.device, dtype=torch.long)
            labels = F.one_hot(labels.long(), num_classes=11).to(dtype=x.dtype)
        elif labels is not None:
            raise ValueError("An unconditional model does not accept class labels")
        data2 = self.sigma_data ** 2
        c_skip = data2 / (data2 + sigma.square())
        c_out = self.sigma_data * sigma / (data2 + sigma.square()).sqrt()
        c_in = (data2 + sigma.square()).rsqrt()
        c_noise = sigma.log() / 4
        prediction = self.net(c_in * x, c_noise.flatten(), labels)
        return c_skip * x + c_out * prediction


def edm_loss(model, clean, labels=None, mean=0.0, std=1.0,
             uncond_prob=0.1, generator=None):
    """Weighted denoising MSE. clean is in [-1,1]; labels are integer IDs."""
    batch = clean.shape[0]
    sigma = (mean + std * torch.randn(
        (batch, 1, 1, 1), device=clean.device, generator=generator
    )).exp()
    noise = torch.randn(clean.shape, device=clean.device, generator=generator) * sigma
    if model.conditional and labels is not None:
        # Do not modify the DataLoader's labels in-place.
        mask = torch.rand(batch, device=clean.device, generator=generator) < uncond_prob
        labels = torch.where(mask, 10, labels)
    denoised = model(clean + noise, sigma, labels)
    weight = (sigma.square() + model.sigma_data ** 2) / (
        sigma * model.sigma_data
    ).square()
    return (weight * (denoised - clean).square()).mean()


def get_timesteps(sigma_min=0.02, sigma_max=80.0, steps=50, rho=7.0, device="cpu"):
    if steps < 2 or not 0 < sigma_min < sigma_max or rho <= 0:
        raise ValueError("Need steps >= 2, 0 < sigma_min < sigma_max, rho > 0")
    ramp = torch.linspace(0, 1, steps, device=device)
    levels = (sigma_max ** (1 / rho) + ramp * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
    )) ** rho
    return torch.cat((levels, levels.new_zeros(1)))


@torch.inference_mode()
def sample_euler(model, count=40, labels=None, steps=50, sigma_min=0.02,
                 sigma_max=80.0, rho=7.0, cfg=0.0, stochastic=False, seed=42):
    """Return raw images in model coordinates (approximately [-1,1]).

    cfg=0 means ordinary conditional sampling, as in the notebook.
    ODE: dx/dsigma = (x-D)/sigma; the sigma step is negative.
    SDE: double the drift and add sqrt(2*sigma*abs(delta_sigma))*N(0,I).
    """
    if count < 1 or cfg < 0:
        raise ValueError("count must be positive and cfg must be nonnegative")
    if cfg and (not model.conditional or labels is None):
        raise ValueError("CFG requires a conditional model and explicit digit labels")
    device = next(model.parameters()).device
    if labels is not None:
        labels = torch.as_tensor(labels, device=device, dtype=torch.long)
        if labels.shape != (count,) or not torch.all((labels >= 0) & (labels <= 10)):
            raise ValueError("Provide count labels in the range 0..10")
    model.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((count, 1, 16, 16), device=device, generator=rng) * sigma_max
    levels = get_timesteps(sigma_min, sigma_max, steps, rho, device)
    for current, following in zip(levels[:-1], levels[1:]):
        denoised = model(x, current, labels)
        if cfg:
            uncond = model(x, current, torch.full_like(labels, 10))
            denoised = denoised + cfg * (denoised - uncond)
        drift = (x - denoised) / current
        delta = following - current
        x = x + (2 if stochastic else 1) * drift * delta
        if stochastic:
            x = x + (2 * current * delta.abs()).sqrt() * torch.randn(
                x.shape, device=device, generator=rng
            )
    return x


def save_samples(images, path, columns=10):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Clamp only for display; never feed clipped/normalized pictures to the solver.
    save_image((images.detach().cpu() * 0.5 + 0.5).clamp(0, 1), path, nrow=columns)


@torch.no_grad()
def calc_data_std(loader):
    """Match the notebook: sqrt(mean over pixels of Var over images))."""
    total = torch.zeros(1, 16, 16, dtype=torch.float64)
    total2 = torch.zeros_like(total)
    count = 0
    for images, _ in loader:
        images = images.double()
        total += images.sum(0)
        total2 += images.square().sum(0)
        count += len(images)
    variance = (total2 / count - (total / count).square()).mean()
    return variance.clamp_min(0).sqrt().item()


def make_loaders(data_dir, batch_size, workers, device):
    transform = Compose([Resize((16, 16)), ToTensor()])
    train = MNIST(data_dir, train=True, transform=transform, download=True)
    test = MNIST(data_dir, train=False, transform=transform, download=True)
    kwargs = dict(batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda")
    return DataLoader(train, shuffle=True, **kwargs), DataLoader(test, shuffle=False, **kwargs)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    # Fixed noise makes epoch-to-epoch test loss easier to compare.
    rng = torch.Generator(device=device).manual_seed(12345)
    total = 0.0
    count = 0
    for images, labels in loader:
        clean = images.to(device) * 2 - 1
        labels = labels.to(device) if model.conditional else None
        loss = edm_loss(model, clean, labels, uncond_prob=0.0, generator=rng)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite test loss")
        total += loss.item() * len(images)
        count += len(images)
    return total / count


def save_checkpoint(model, path, epoch=0, optimizer=None, metrics=None):
    """Only tensors and simple metadata: load with weights_only=True."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    checkpoint = dict(format_version=1, model_config=model.config, model_state=state,
                      epoch=epoch, metrics=metrics or {})
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model = EDMPrecond(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval()


def train(args, device):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pth").exists():
        raise FileExistsError("Output already contains last.pth; choose a new --output directory")
    train_loader, test_loader = make_loaders(args.data_dir, args.batch_size, args.workers, device)
    sigma_data = calc_data_std(train_loader) * 2
    model = EDMPrecond(sigma_data, args.conditional).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; "
          f"sigma_data: {sigma_data:.6f}", flush=True)
    config = {**vars(args), "device": str(device), "sigma_data": sigma_data,
              "torch_version": str(torch.__version__)}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best = math.inf
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "test_loss"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            total, seen = 0.0, 0
            for step, (images, labels) in enumerate(train_loader, start=1):
                clean = images.to(device) * 2 - 1
                labels = labels.to(device) if model.conditional else None
                optimizer.zero_grad(set_to_none=True)
                loss = edm_loss(model, clean, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                total += loss.item() * len(images)
                seen += len(images)
                if step == 1 or step % 50 == 0:
                    print(f"Epoch {epoch}/{args.epochs}, batch {step}/{len(train_loader)}, "
                          f"loss={loss.item():.5f}", flush=True)
            metrics = dict(epoch=epoch, train_loss=total / seen,
                           test_loss=evaluate(model, test_loader, device))
            writer.writerow(metrics)
            stream.flush()
            save_checkpoint(model, output / "last.pth", epoch, optimizer, metrics)
            if metrics["test_loss"] < best:
                best = metrics["test_loss"]
                save_checkpoint(model, output / "best.pth", epoch, metrics=metrics)
            labels = torch.arange(40, device=device) % 10 if model.conditional else None
            samples = sample_euler(model, count=40, labels=labels, seed=args.seed)
            save_samples(samples, output / f"epoch_{epoch:03d}.png")
            print(f"Epoch {epoch}: train={metrics['train_loss']:.5f}, "
                  f"test={metrics['test_loss']:.5f}; saved to {output}", flush=True)


def generate(args, device):
    model = load_checkpoint(args.checkpoint, device)
    labels = None
    if model.conditional:
        if args.digit is not None:
            labels = torch.full((args.count,), args.digit, device=device)
        elif args.unconditional:
            labels = torch.full((args.count,), 10, device=device)
        else:
            # Each row contains 0,1,...,9 when the grid has 10 columns.
            labels = torch.arange(args.count, device=device) % 10
    elif args.digit is not None or args.unconditional or args.cfg:
        raise ValueError("Class selection and CFG require a conditional checkpoint")
    if args.unconditional and args.cfg:
        raise ValueError("--unconditional cannot be combined with nonzero --cfg")
    images = sample_euler(
        model, count=args.count, labels=labels, steps=args.steps,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max, rho=args.rho,
        cfg=args.cfg, stochastic=args.stochastic, seed=args.seed,
    )
    if not torch.isfinite(images).all():
        raise FloatingPointError("Sampling produced non-finite values")
    save_samples(images, args.output)
    metadata = {**vars(args), "device": str(device),
                "labels": labels.cpu().tolist() if labels is not None else None}
    Path(args.output).with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved {args.count} images: {args.output}")


def smoke_test(device):
    """Synthetic checks only; no MNIST download and no meaningful training."""
    levels = get_timesteps(device=device)
    assert levels[-1] == 0 and torch.all(levels[:-1] > levels[1:])
    for conditional in (False, True):
        model = EDMPrecond(0.4144, conditional).to(device)
        clean = torch.rand(2, 1, 16, 16, device=device) * 2 - 1
        labels = torch.tensor([2, 7], device=device) if conditional else None
        original_labels = labels.clone() if conditional else None
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        model.train()
        loss = edm_loss(model, clean, labels, uncond_prob=1.0)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        if conditional:
            assert torch.equal(labels, original_labels)
        model.eval()
        kwargs = dict(count=2, labels=labels, steps=3, sigma_max=1.0, seed=17)
        before = sample_euler(model, **kwargs)
        assert before.shape == clean.shape and torch.isfinite(before).all()
        assert torch.equal(before, sample_euler(model, **kwargs))
        stoch = sample_euler(model, **kwargs, stochastic=True)
        assert torch.isfinite(stoch).all() and not torch.equal(before, stoch)
        if conditional:
            guided = sample_euler(model, **kwargs, cfg=2.0)
            assert torch.isfinite(guided).all()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            save_checkpoint(model, path, optimizer=optimizer)
            restored = load_checkpoint(path, device)
            assert torch.equal(before, sample_euler(restored, **kwargs))
            save_samples(before, Path(directory) / "samples.png")
        print(f"PASS: conditional={conditional}; parameters="
              f"{sum(p.numel() for p in model.parameters()):,}; loss={loss.item():.5f}", flush=True)
    # Analytic oracle: D(x)=0 => one exact final ODE step should give x=0.
    class ZeroDenoiser(nn.Module):
        conditional = False

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros((), device=device))

        def forward(self, x, sigma, labels=None):
            return torch.zeros_like(x)

    assert sample_euler(ZeroDenoiser(), count=2, steps=3).abs().max() < 1e-6
    print("PASS: sampler sign and final step; all smoke checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--threads", type=int, default=4, help="CPU threads, default: 4")
    training = commands.add_parser("train", parents=[common], help="Train on MNIST")
    training.add_argument("--conditional", action="store_true")
    training.add_argument("--epochs", type=int, default=6)
    training.add_argument("--batch-size", type=int, default=256)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--workers", type=int, default=0, help="0 is convenient on Windows")
    training.add_argument("--data-dir", default="data")
    training.add_argument("--output", default="runs/mnist")
    sampling = commands.add_parser("sample", parents=[common], help="Generate a PNG grid")
    sampling.add_argument("--checkpoint", required=True)
    selection = sampling.add_mutually_exclusive_group()
    selection.add_argument("--digit", type=int, choices=range(10))
    selection.add_argument("--unconditional", action="store_true")
    sampling.add_argument("--count", type=int, default=40)
    sampling.add_argument("--steps", type=int, default=50)
    sampling.add_argument("--sigma-min", type=float, default=0.02)
    sampling.add_argument("--sigma-max", type=float, default=80.0)
    sampling.add_argument("--rho", type=float, default=7.0)
    sampling.add_argument("--cfg", type=float, default=0.0)
    sampling.add_argument("--stochastic", action="store_true")
    sampling.add_argument("--output", default="samples.png")
    commands.add_parser("smoke-test", parents=[common], help="Check with synthetic tensors")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.command == "train" and (args.epochs < 1 or args.batch_size < 1 or
                                    args.workers < 0 or not args.lr > 0):
        parser.error("epochs, batch-size and lr must be positive; workers must be >= 0")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu or install a CUDA build of PyTorch.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if args.command == "train":
        train(args, device)
    elif args.command == "sample":
        generate(args, device)
    else:
        smoke_test(device)


if __name__ == "__main__":
    main()
