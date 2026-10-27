"""Formula and integration checks; run: python -m unittest -v test_diffusion.py."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import diffusion_mnist as dm


class ConstantDenoiser(nn.Module):
    conditional = True

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, sigma, labels=None):
        value = torch.where(labels == 10, 0.1, 0.4).reshape(-1, 1, 1, 1)
        return value.expand_as(x)


class DiffusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(42)

    def test_cfg_sign_and_last_step(self):
        # A constant denoiser gives an analytic final Euler output.
        model = ConstantDenoiser()
        labels = torch.tensor([3, 7])
        for cfg, expected in [(0, 0.4), (2, 1.0)]:
            result = dm.sample_euler(model, count=2, labels=labels, cfg=cfg, steps=3)
            torch.testing.assert_close(result, torch.full_like(result, expected))

    def test_schedule_and_invalid_parameters(self):
        levels = dm.get_timesteps()
        self.assertEqual(len(levels), 51)
        self.assertAlmostEqual(levels[0].item(), 80, places=3)
        self.assertAlmostEqual(levels[-2].item(), 0.02, places=6)
        self.assertEqual(levels[-1].item(), 0)
        self.assertTrue(torch.all(levels[:-1] > levels[1:]))
        for kwargs in [dict(steps=1), dict(sigma_min=0), dict(rho=0)]:
            with self.assertRaises(ValueError):
                dm.get_timesteps(**kwargs)

    def test_data_std_matches_pixelwise_definition(self):
        images = torch.stack([torch.zeros(1, 16, 16), torch.ones(1, 16, 16)])
        loader = DataLoader(TensorDataset(images, torch.tensor([0, 1])), batch_size=1)
        self.assertEqual(dm.calc_data_std(loader), 0.5)

    def test_one_epoch_training_and_sample_cli_handler(self):
        # Exercise the real train/evaluate/checkpoint path on a tiny synthetic dataset.
        # Only shorten the preview generation; the sampler itself is still real.
        images = torch.rand(2, 1, 16, 16)
        labels = torch.tensor([1, 7])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        original_sampler = dm.sample_euler

        def short_preview(model, **kwargs):
            kwargs.update(count=2, steps=2, sigma_max=1.0)
            kwargs['labels'] = kwargs['labels'][:2]
            return original_sampler(model, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=directory, data_dir='unused', batch_size=2,
                                      workers=0, lr=3e-4, conditional=True, epochs=1, seed=42)
            with patch.object(dm, 'make_loaders', return_value=(loader, loader)), \
                    patch.object(dm, 'sample_euler', side_effect=short_preview):
                dm.train(args, torch.device('cpu'))
            for filename in ['last.pth', 'best.pth', 'metrics.csv', 'config.json', 'epoch_001.png']:
                self.assertTrue((Path(directory) / filename).is_file(), filename)
            generation = argparse.Namespace(
                checkpoint=str(Path(directory) / 'best.pth'), digit=7, unconditional=False,
                count=2, steps=2, sigma_min=0.02, sigma_max=1.0, rho=7.0,
                cfg=2.0, stochastic=False, seed=42, output=str(Path(directory) / 'digits.png'),
            )
            dm.generate(generation, torch.device('cpu'))
            self.assertTrue(Path(generation.output).is_file())
            self.assertTrue(Path(generation.output).with_suffix('.json').is_file())


if __name__ == '__main__':
    unittest.main()
