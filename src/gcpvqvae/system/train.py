"""High level training loop orchestration."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from gcpvqvae.data.dataset import BackboneDataset, collate_backbones
from gcpvqvae.models.gcpvqvae_model import GCPVQVAE
from gcpvqvae.utils.checkpoint import save_checkpoint
from gcpvqvae.utils.logging import Logger


def get_cosine_schedule_with_warmup(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """Simple cosine learning rate scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class Trainer:
    """Orchestrates the training process."""
    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model
        self.model = GCPVQVAE(config['model']).to(self.device)

        # Optimizer and Scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=config['train']['lr'])
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['train']['warmup_steps'],
            num_training_steps=config['train']['total_steps']
        )

        # Dataloader
        dataset = BackboneDataset(**config['data'])
        self.dataloader = DataLoader(
            dataset,
            batch_size=config['train']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            collate_fn=collate_backbones,
        )

        # AMP
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['train']['use_amp'])

        # Logging
        self.logger = Logger(log_dir=Path(config['train']['save_path']).parent)
        self.step = 0

    def train(self):
        """Main training loop."""
        self.model.train()

        pbar = tqdm(total=self.config['train']['total_steps'])
        while self.step < self.config['train']['total_steps']:
            for batch in self.dataloader:
                if batch is None: continue

                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                with torch.cuda.amp.autocast(enabled=self.config['train']['use_amp']):
                    output = self.model(batch)
                    loss = output['loss']

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['clip_grad_norm'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                pbar.update(1)
                self.step += 1

                if self.step % 100 == 0:
                    log_metrics = {k: v.item() for k, v in output.items() if 'loss' in k}
                    log_metrics['lr'] = self.scheduler.get_last_lr()[0]
                    self.logger.log_metrics(log_metrics, self.step, prefix="train/")

                if self.step % 1000 == 0:
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.step,
                        self.config,
                        self.config['train']['save_path']
                    )

                if self.step >= self.config['train']['total_steps']:
                    break

        self.logger.close()
        print("Training finished.")

        # Save final checkpoint
        save_checkpoint(
            self.model,
            self.optimizer,
            self.step,
            self.config,
            self.config['train']['save_path']
        )


def train_from_config(config_path: str):
    """Load config and run training."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    trainer.train()
