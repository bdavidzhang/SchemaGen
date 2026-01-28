"""
Training Pipeline for SchemaGNN.

Handles:
- Dataset creation from corrupted schemas
- Training loop with dual loss heads
- Evaluation metrics
- Model checkpointing
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import HeteroData
from tqdm import tqdm

from .model import SchemaGNN, SchemaGNNLoss
from .parser import SchemaGraphParser
from .corruptor import SchemaCorruptor


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 50
    warmup_epochs: int = 5
    
    # Loss weights
    global_loss_weight: float = 1.0
    local_loss_weight: float = 0.5
    
    # Data
    valid_ratio: float = 0.3
    corruptions_per_schema: int = 3
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5


@dataclass 
class TrainingMetrics:
    """Metrics from a training epoch."""
    epoch: int
    train_loss: float
    train_global_loss: float
    train_local_loss: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    val_node_accuracy: Optional[float] = None


class SchemaDataset:
    """
    Dataset class for training SchemaGNN.
    
    Converts corrupted schemas into HeteroData objects with labels.
    """
    
    def __init__(
        self,
        examples: list[dict],
        parser: SchemaGraphParser,
    ):
        """
        Initialize dataset.
        
        Args:
            examples: List of dicts with 'schema', 'is_valid', 'corrupted_paths'
            parser: SchemaGraphParser instance
        """
        self.examples = examples
        self.parser = parser
        self._cache: dict[int, HeteroData] = {}
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> HeteroData:
        if idx in self._cache:
            return self._cache[idx]
            
        example = self.examples[idx]
        
        try:
            # Parse schema to graph
            data = self.parser.parse(example["schema"])
            
            # Add labels
            data.y = torch.tensor([1.0 if example["is_valid"] else 0.0])
            
            # Create node-level labels
            num_nodes = data["schema_node"].num_nodes
            node_labels = torch.zeros(num_nodes)
            
            # Mark corrupted nodes
            corrupted_paths = set(example.get("corrupted_paths", []))
            for i, path in enumerate(data.node_paths):
                # Check if this node or any parent is corrupted
                for corrupted in corrupted_paths:
                    if path.startswith(corrupted) or corrupted.startswith(path):
                        node_labels[i] = 1.0
                        break
                        
            data.node_y = node_labels
            
            self._cache[idx] = data
            return data
            
        except Exception as e:
            # Return a dummy valid graph on parse error
            print(f"Warning: Failed to parse example {idx}: {e}")
            dummy = HeteroData()
            dummy["schema_node"].x = torch.zeros(1, 404)
            dummy["schema_node"].num_nodes = 1
            dummy.y = torch.tensor([1.0])
            dummy.node_y = torch.zeros(1)
            dummy.node_paths = ["root"]
            dummy.node_types = ["ROOT"]
            return dummy


class Trainer:
    """
    Training pipeline for SchemaGNN.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        device: str = "cpu",
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            device: Device to train on
        """
        self.config = config
        self.device = device
        
        # Will be initialized in setup()
        self.model: Optional[SchemaGNN] = None
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[CosineAnnealingLR] = None
        self.criterion: Optional[SchemaGNNLoss] = None
        self.parser: Optional[SchemaGraphParser] = None
        
    def setup(self, input_dim: int = 404) -> None:
        """Initialize model, optimizer, and criterion."""
        self.model = SchemaGNN(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
        )
        
        self.criterion = SchemaGNNLoss(
            global_weight=self.config.global_loss_weight,
            local_weight=self.config.local_loss_weight,
        )
        
        self.parser = SchemaGraphParser(device=self.device)
        
    def prepare_data(
        self,
        schemas: list[dict],
        val_split: float = 0.1,
    ) -> tuple[TorchDataLoader, Optional[TorchDataLoader]]:
        """
        Prepare training and validation data loaders.
        
        Args:
            schemas: List of valid JSON schemas
            val_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Generate corrupted examples
        corruptor = SchemaCorruptor(seed=42)
        examples = corruptor.generate_dataset(
            schemas,
            valid_ratio=self.config.valid_ratio,
            corruptions_per_schema=self.config.corruptions_per_schema,
        )
        
        # Split data
        split_idx = int(len(examples) * (1 - val_split))
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        # Create datasets
        train_dataset = SchemaDataset(train_examples, self.parser)
        val_dataset = SchemaDataset(val_examples, self.parser) if val_examples else None
        
        # Create data loaders using standard PyTorch DataLoader
        # HeteroData batching requires special handling, so we process one at a time
        def collate_fn(batch):
            # batch is a list of HeteroData objects, just return it
            return batch
            
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=1,  # Process one graph at a time
            shuffle=True,
            collate_fn=collate_fn,
        )
        
        val_loader = None
        if val_dataset:
            val_loader = TorchDataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn,
            )
            
        return train_loader, val_loader
        
    def train_epoch(
        self,
        train_loader: TorchDataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_global_loss = 0.0
        total_local_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            for data in batch:  # Each batch is a list of HeteroData
                data = data.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                
                # Compute loss
                losses = self.criterion(
                    output,
                    data.y,
                    data.node_y,
                )
                
                # Backward pass
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += losses["total"].item()
                total_global_loss += losses["global"].item()
                total_local_loss += losses["local"].item()
                num_batches += 1
                
                pbar.set_postfix({
                    "loss": f"{losses['total'].item():.4f}",
                    "global": f"{losses['global'].item():.4f}",
                    "local": f"{losses['local'].item():.4f}",
                })
                
        return {
            "loss": total_loss / max(num_batches, 1),
            "global_loss": total_global_loss / max(num_batches, 1),
            "local_loss": total_local_loss / max(num_batches, 1),
        }
        
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: TorchDataLoader,
    ) -> dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        correct_global = 0
        correct_local = 0
        total_global = 0
        total_local = 0
        
        for batch in val_loader:
            for data in batch:
                data = data.to(self.device)
                
                output = self.model(data)
                
                losses = self.criterion(
                    output,
                    data.y,
                    data.node_y,
                )
                
                total_loss += losses["total"].item()
                
                # Global accuracy
                pred_valid = output["validity_score"] >= 0.5
                correct_global += (pred_valid == data.y.bool()).sum().item()
                total_global += 1
                
                # Node accuracy
                pred_errors = output["node_error_probs"] >= 0.5
                correct_local += (pred_errors == data.node_y.bool()).sum().item()
                total_local += data.node_y.numel()
                
        return {
            "loss": total_loss / max(total_global, 1),
            "accuracy": correct_global / max(total_global, 1),
            "node_accuracy": correct_local / max(total_local, 1),
        }
        
    def save_checkpoint(self, epoch: int, metrics: dict) -> Path:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"schema_gnn_epoch_{epoch}.pt"
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }, checkpoint_path)
        
        return checkpoint_path
        
    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load model from checkpoint. Returns the epoch number."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return checkpoint["epoch"]
        
    def train(
        self,
        schemas: list[dict],
        val_split: float = 0.1,
    ) -> list[TrainingMetrics]:
        """
        Full training loop.
        
        Args:
            schemas: List of valid JSON schemas for training
            val_split: Fraction for validation
            
        Returns:
            List of TrainingMetrics for each epoch
        """
        if self.model is None:
            self.setup()
            
        train_loader, val_loader = self.prepare_data(schemas, val_split)
        
        history = []
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_metrics = None
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                
            # Update scheduler
            self.scheduler.step()
            
            # Record metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_global_loss=train_metrics["global_loss"],
                train_local_loss=train_metrics["local_loss"],
                val_loss=val_metrics["loss"] if val_metrics else None,
                val_accuracy=val_metrics["accuracy"] if val_metrics else None,
                val_node_accuracy=val_metrics["node_accuracy"] if val_metrics else None,
            )
            history.append(metrics)
            
            # Log progress
            log_msg = f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}"
            if val_metrics:
                log_msg += f", val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.2%}"
            print(log_msg)
            
            # Save checkpoint
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(epoch, train_metrics)
                
        # Save final model
        self.save_checkpoint(self.config.num_epochs, train_metrics)
        
        return history


def train_from_schema_files(
    schema_paths: list[Path],
    config: Optional[TrainingConfig] = None,
    device: str = "cpu",
) -> Trainer:
    """
    Convenience function to train from schema files.
    
    Args:
        schema_paths: Paths to JSON schema files
        config: Training configuration
        device: Device to train on
        
    Returns:
        Trained Trainer instance
    """
    if config is None:
        config = TrainingConfig()
        
    # Load schemas
    schemas = []
    for path in schema_paths:
        with open(path) as f:
            schemas.append(json.load(f))
            
    # Train
    trainer = Trainer(config, device)
    trainer.setup()
    trainer.train(schemas)
    
    return trainer
