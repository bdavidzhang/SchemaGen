"""
SchemaGNN: Heterogeneous Graph Transformer for JSON Schema validation.

Uses an HGT backbone with two prediction heads:
1. Global Critic Head: Binary classification (valid/invalid)
2. Local Debugger Head: Node-level error detection
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, GCNConv, GATConv, Linear, global_mean_pool, global_max_pool


class SchemaGNN(nn.Module):
    """
    Heterogeneous Graph Transformer for validating JSON Schemas.
    
    Architecture:
    - Input projection layer
    - 3-4 HGT message passing layers
    - Global Critic Head (graph-level binary classification)
    - Local Debugger Head (node-level error detection)
    """
    
    def __init__(
        self,
        input_dim: int = 404,  # 384 semantic + 11 type + 1 depth + 8 constraints
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize SchemaGNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads in HGT
            num_layers: Number of HGT layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Node type metadata (single type for now - can extend)
        self.node_types = ["schema_node"]
        
        # Edge type metadata
        self.edge_types = [
            ("schema_node", "contains", "schema_node"),
            ("schema_node", "items", "schema_node"),
            ("schema_node", "refers_to", "schema_node"),
            ("schema_node", "logic", "schema_node"),
            ("schema_node", "additional", "schema_node"),
        ]
        
        self.metadata = (self.node_types, self.edge_types)
        
        # Input projection
        self.input_proj = Linear(input_dim, hidden_dim)
        
        # HGT layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=self.metadata,
                heads=num_heads,
            )
            self.convs.append(conv)
            
        # Layer normalization
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Global Critic Head (graph-level classification)
        self.global_head = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, 1),
        )
        
        # Local Debugger Head (node-level error detection)
        self.local_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, 1),
        )
        
    def forward(
        self,
        data: HeteroData,
        return_node_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            data: HeteroData object from SchemaGraphParser
            return_node_embeddings: If True, include final node embeddings in output
            
        Returns:
            Dictionary with:
            - 'validity_score': (batch_size,) global validity probability
            - 'node_error_probs': (num_nodes,) per-node error probability
            - 'node_embeddings': (num_nodes, hidden_dim) if return_node_embeddings
        """
        # Get input features
        x_dict = {"schema_node": data["schema_node"].x}
        
        # Build edge_index dict (filter out empty edge types)
        edge_index_dict = {}
        for edge_type in self.edge_types:
            edge_key = edge_type
            if edge_key in data and data[edge_key].edge_index.numel() > 0:
                edge_index_dict[edge_key] = data[edge_key].edge_index
                
        # Input projection
        x_dict["schema_node"] = self.input_proj(x_dict["schema_node"])
        x_dict["schema_node"] = F.relu(x_dict["schema_node"])
        
        # HGT message passing
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Handle case where no edges exist
            if edge_index_dict:
                x_new = conv(x_dict, edge_index_dict)
                x_dict["schema_node"] = norm(
                    x_dict["schema_node"] + self.dropout(x_new["schema_node"])
                )
            else:
                # No edges - just apply norm
                x_dict["schema_node"] = norm(x_dict["schema_node"])
                
        node_embeddings = x_dict["schema_node"]
        
        # Get batch assignment (for batched graphs)
        if hasattr(data["schema_node"], "batch"):
            batch = data["schema_node"].batch
        else:
            batch = torch.zeros(node_embeddings.size(0), dtype=torch.long, 
                              device=node_embeddings.device)
            
        # Global Critic Head
        # Combine mean and max pooling for richer representation
        graph_embed_mean = global_mean_pool(node_embeddings, batch)
        graph_embed_max = global_max_pool(node_embeddings, batch)
        graph_embed = torch.cat([graph_embed_mean, graph_embed_max], dim=-1)
        validity_logits = self.global_head(graph_embed).squeeze(-1)
        validity_score = torch.sigmoid(validity_logits)
        
        # Local Debugger Head
        node_error_logits = self.local_head(node_embeddings).squeeze(-1)
        node_error_probs = torch.sigmoid(node_error_logits)
        
        output = {
            "validity_score": validity_score,
            "validity_logits": validity_logits,
            "node_error_probs": node_error_probs,
            "node_error_logits": node_error_logits,
        }
        
        if return_node_embeddings:
            output["node_embeddings"] = node_embeddings
            
        return output
        
    def get_defect_nodes(
        self,
        data: HeteroData,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Get the nodes most likely to contain errors.
        
        Args:
            data: HeteroData object
            threshold: Probability threshold for error detection
            top_k: If set, return top-k highest probability nodes
            
        Returns:
            List of dicts with node_id, json_path, probability, node_type
        """
        self.eval()
        with torch.no_grad():
            output = self(data)
            
        probs = output["node_error_probs"].cpu().numpy()
        
        # Get defect nodes
        defects = []
        if top_k is not None:
            # Get top-k indices
            indices = probs.argsort()[-top_k:][::-1]
        else:
            # Get all above threshold
            indices = (probs >= threshold).nonzero()[0]
            
        for idx in indices:
            defects.append({
                "node_id": int(idx),
                "json_path": data.node_paths[idx] if hasattr(data, "node_paths") else f"node_{idx}",
                "node_type": data.node_types[idx] if hasattr(data, "node_types") else "unknown",
                "probability": float(probs[idx]),
            })
            
        # Sort by probability descending
        defects.sort(key=lambda x: x["probability"], reverse=True)
        return defects


class SchemaGNNLoss(nn.Module):
    """
    Combined loss function for SchemaGNN training.
    
    Combines:
    - Binary Cross Entropy for global validity prediction
    - Binary Cross Entropy for node-level error detection
    """
    
    def __init__(
        self,
        global_weight: float = 1.0,
        local_weight: float = 1.0,
        pos_weight_global: Optional[float] = None,
        pos_weight_local: Optional[float] = None,
    ):
        """
        Initialize loss function.
        
        Args:
            global_weight: Weight for global validity loss
            local_weight: Weight for local node error loss
            pos_weight_global: Positive class weight for global loss (for imbalance)
            pos_weight_local: Positive class weight for local loss
        """
        super().__init__()
        self.global_weight = global_weight
        self.local_weight = local_weight
        
        self.global_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_global]) if pos_weight_global else None
        )
        self.local_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_local]) if pos_weight_local else None,
            reduction="none",  # For handling masks
        )
        
    def forward(
        self,
        output: dict[str, torch.Tensor],
        validity_target: torch.Tensor,
        node_error_target: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            output: Model output dict
            validity_target: (batch_size,) binary validity labels
            node_error_target: (num_nodes,) binary node error labels
            node_mask: Optional mask for nodes with labels
            
        Returns:
            Dict with 'total', 'global', and 'local' losses
        """
        # Global validity loss
        global_loss = self.global_criterion(
            output["validity_logits"], 
            validity_target.float()
        )
        
        # Local node error loss
        local_losses = self.local_criterion(
            output["node_error_logits"],
            node_error_target.float()
        )
        
        if node_mask is not None:
            local_loss = (local_losses * node_mask).sum() / node_mask.sum().clamp(min=1)
        else:
            local_loss = local_losses.mean()
            
        # Combined loss
        total_loss = (
            self.global_weight * global_loss + 
            self.local_weight * local_loss
        )
        
        return {
            "total": total_loss,
            "global": global_loss,
            "local": local_loss,
        }


class HomogeneousSchemaGNN(nn.Module):
    """
    Homogeneous GNN baseline for ablation studies.
    
    Unlike SchemaGNN which uses HGT (Heterogeneous Graph Transformer),
    this model treats all nodes and edges the same, using either GCN or GAT.
    This helps quantify the value of heterogeneous modeling.
    
    Architecture:
    - Input projection layer
    - 3-4 GCN/GAT message passing layers
    - Global Critic Head (graph-level binary classification)
    - Local Debugger Head (node-level error detection)
    """
    
    def __init__(
        self,
        input_dim: int = 404,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        conv_type: str = "gcn",  # "gcn" or "gat"
        num_heads: int = 4,  # Only used for GAT
    ):
        """
        Initialize HomogeneousSchemaGNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            dropout: Dropout probability
            conv_type: Type of convolution ("gcn" or "gat")
            num_heads: Number of attention heads (for GAT only)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.num_heads = num_heads
        
        # Input projection
        self.input_proj = Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if conv_type == "gat":
                # For GAT, handle multi-head output dimensions
                in_dim = hidden_dim if i == 0 else hidden_dim
                conv = GATConv(
                    in_channels=in_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                )
            else:  # gcn
                conv = GCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                )
            self.convs.append(conv)
            
        # Layer normalization
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Global Critic Head (graph-level classification)
        self.global_head = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, 1),
        )
        
        # Local Debugger Head (node-level error detection)
        self.local_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, 1),
        )
        
    def _collect_edges(self, data: HeteroData) -> torch.Tensor:
        """Collect all edges from heterogeneous data into a single edge_index."""
        edge_types = [
            ("schema_node", "contains", "schema_node"),
            ("schema_node", "items", "schema_node"),
            ("schema_node", "refers_to", "schema_node"),
            ("schema_node", "logic", "schema_node"),
            ("schema_node", "additional", "schema_node"),
        ]
        
        all_edges = []
        for edge_type in edge_types:
            if edge_type in data and data[edge_type].edge_index.numel() > 0:
                all_edges.append(data[edge_type].edge_index)
                
        if all_edges:
            return torch.cat(all_edges, dim=1)
        else:
            # Return empty edge index
            return torch.zeros((2, 0), dtype=torch.long, device=data["schema_node"].x.device)
        
    def forward(
        self,
        data: HeteroData,
        return_node_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            data: HeteroData object from SchemaGraphParser
            return_node_embeddings: If True, include final node embeddings in output
            
        Returns:
            Dictionary with:
            - 'validity_score': (batch_size,) global validity probability
            - 'node_error_probs': (num_nodes,) per-node error probability
            - 'node_embeddings': (num_nodes, hidden_dim) if return_node_embeddings
        """
        # Get input features
        x = data["schema_node"].x
        
        # Collect all edges (treating them homogeneously)
        edge_index = self._collect_edges(data)
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GNN message passing
        for conv, norm in zip(self.convs, self.norms):
            if edge_index.numel() > 0:
                x_new = conv(x, edge_index)
                x = norm(x + self.dropout(x_new))
            else:
                x = norm(x)
                
        node_embeddings = x
        
        # Get batch assignment (for batched graphs)
        if hasattr(data["schema_node"], "batch"):
            batch = data["schema_node"].batch
        else:
            batch = torch.zeros(node_embeddings.size(0), dtype=torch.long, 
                              device=node_embeddings.device)
            
        # Global Critic Head
        graph_embed_mean = global_mean_pool(node_embeddings, batch)
        graph_embed_max = global_max_pool(node_embeddings, batch)
        graph_embed = torch.cat([graph_embed_mean, graph_embed_max], dim=-1)
        validity_logits = self.global_head(graph_embed).squeeze(-1)
        validity_score = torch.sigmoid(validity_logits)
        
        # Local Debugger Head
        node_error_logits = self.local_head(node_embeddings).squeeze(-1)
        node_error_probs = torch.sigmoid(node_error_logits)
        
        output = {
            "validity_score": validity_score,
            "validity_logits": validity_logits,
            "node_error_probs": node_error_probs,
            "node_error_logits": node_error_logits,
        }
        
        if return_node_embeddings:
            output["node_embeddings"] = node_embeddings
            
        return output
        
    def get_defect_nodes(
        self,
        data: HeteroData,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Get the nodes most likely to contain errors."""
        self.eval()
        with torch.no_grad():
            output = self(data)
            
        probs = output["node_error_probs"].cpu().numpy()
        
        defects = []
        if top_k is not None:
            indices = probs.argsort()[-top_k:][::-1]
        else:
            indices = (probs >= threshold).nonzero()[0]
            
        for idx in indices:
            defects.append({
                "node_id": int(idx),
                "json_path": data.node_paths[idx] if hasattr(data, "node_paths") else f"node_{idx}",
                "node_type": data.node_types[idx] if hasattr(data, "node_types") else "unknown",
                "probability": float(probs[idx]),
            })
            
        defects.sort(key=lambda x: x["probability"], reverse=True)
        return defects


class NoSemanticSchemaGNN(nn.Module):
    """
    SchemaGNN variant without semantic embeddings for ablation.
    
    This model only uses structural features (type one-hot, depth, constraints)
    without the 384-dim semantic embedding from MiniLM. This helps quantify
    the value of semantic information.
    """
    
    def __init__(
        self,
        input_dim: int = 20,  # 11 type + 1 depth + 8 constraints (no 384 semantic)
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize NoSemanticSchemaGNN.
        
        Note: input_dim should be 20 (404 - 384 semantic embedding dims).
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.node_types = ["schema_node"]
        self.edge_types = [
            ("schema_node", "contains", "schema_node"),
            ("schema_node", "items", "schema_node"),
            ("schema_node", "refers_to", "schema_node"),
            ("schema_node", "logic", "schema_node"),
            ("schema_node", "additional", "schema_node"),
        ]
        
        self.metadata = (self.node_types, self.edge_types)
        
        self.input_proj = Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=self.metadata,
                heads=num_heads,
            )
            self.convs.append(conv)
            
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.global_head = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, 1),
        )
        
        self.local_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, 1),
        )
        
    def forward(
        self,
        data: HeteroData,
        return_node_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass - strips semantic embeddings from input."""
        # Get input features and strip semantic embedding (first 384 dims)
        x = data["schema_node"].x
        if x.size(1) > self.input_dim:
            # Remove semantic embedding (assumed to be first 384 dims)
            x = x[:, 384:]  # Keep only structural features
            
        x_dict = {"schema_node": x}
        
        edge_index_dict = {}
        for edge_type in self.edge_types:
            if edge_type in data and data[edge_type].edge_index.numel() > 0:
                edge_index_dict[edge_type] = data[edge_type].edge_index
                
        x_dict["schema_node"] = self.input_proj(x_dict["schema_node"])
        x_dict["schema_node"] = F.relu(x_dict["schema_node"])
        
        for conv, norm in zip(self.convs, self.norms):
            if edge_index_dict:
                x_new = conv(x_dict, edge_index_dict)
                x_dict["schema_node"] = norm(
                    x_dict["schema_node"] + self.dropout(x_new["schema_node"])
                )
            else:
                x_dict["schema_node"] = norm(x_dict["schema_node"])
                
        node_embeddings = x_dict["schema_node"]
        
        if hasattr(data["schema_node"], "batch"):
            batch = data["schema_node"].batch
        else:
            batch = torch.zeros(node_embeddings.size(0), dtype=torch.long, 
                              device=node_embeddings.device)
            
        graph_embed_mean = global_mean_pool(node_embeddings, batch)
        graph_embed_max = global_max_pool(node_embeddings, batch)
        graph_embed = torch.cat([graph_embed_mean, graph_embed_max], dim=-1)
        validity_logits = self.global_head(graph_embed).squeeze(-1)
        validity_score = torch.sigmoid(validity_logits)
        
        node_error_logits = self.local_head(node_embeddings).squeeze(-1)
        node_error_probs = torch.sigmoid(node_error_logits)
        
        output = {
            "validity_score": validity_score,
            "validity_logits": validity_logits,
            "node_error_probs": node_error_probs,
            "node_error_logits": node_error_logits,
        }
        
        if return_node_embeddings:
            output["node_embeddings"] = node_embeddings
            
        return output


def create_model(
    model_type: str = "hgt",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create different model variants.
    
    Args:
        model_type: One of "hgt" (default), "gcn", "gat", "no_semantic"
        **kwargs: Arguments passed to the model constructor
        
    Returns:
        Initialized model
    """
    if model_type == "hgt":
        return SchemaGNN(**kwargs)
    elif model_type == "gcn":
        return HomogeneousSchemaGNN(conv_type="gcn", **kwargs)
    elif model_type == "gat":
        return HomogeneousSchemaGNN(conv_type="gat", **kwargs)
    elif model_type == "no_semantic":
        # Adjust input_dim for no semantic embeddings
        kwargs.pop("input_dim", None)
        return NoSemanticSchemaGNN(input_dim=20, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: hgt, gcn, gat, no_semantic")
