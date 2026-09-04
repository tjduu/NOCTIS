import torch
import torch.nn as nn
import lightning as L
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import grid
import torch.nn.functional as F
from noctis.forecast.loss import PersistenceDefeatingLoss


class GNNForecaster(L.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.save_hyperparameters(config)
        
        in_channels = config.get('in_channels', 24)
        hidden_channels = config.get('hidden_channels', 64)
        out_channels = config.get('out_channels', 24)
        
        # Spatial dimensions (64x64)
        self.h = config.get('in_shape', [24, 1, 64, 64])[2]
        self.w = config.get('in_shape', [24, 1, 64, 64])[3]
        
        # Define the 3 levels of nodes
        self.num_micro = self.h * self.w               # 4096 (Pixels)
        self.num_meso = (self.h // 4) * (self.w // 4)  # 256 (Regional Hubs)
        self.num_macro = (self.h // 16) * (self.w // 16) # 16 (Macro Hubs)
        self.total_nodes = self.num_micro + self.num_meso + self.num_macro # 4368
        
        # Standard GNN Message Passing Layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.act1 = nn.ReLU()
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.act2 = nn.ReLU()
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.act3 = nn.ReLU()
        self.conv4 = SAGEConv(hidden_channels, out_channels)
        
        self.criterion = PersistenceDefeatingLoss()
        
        # Build the static 3-level graph topology once and save it to the GPU
        base_edges = self._build_hierarchical_topology()
        self.register_buffer('base_edge_index', base_edges)
        
        self.cached_batch_size = -1
        self.cached_edge_index = None

    def _build_hierarchical_topology(self):
        """Wires the 3 levels of nodes together."""
        edges = []
        
        # 1. Micro-level edges: Standard 4-neighbor grid (0 to 4095)
        grid_edges, _ = grid(self.h, self.w)
        edges.append(grid_edges)
        
        # 2. Micro-to-Meso edges (Connect 4x4 pixel blocks to their regional hub)
        micro_idx = torch.arange(self.num_micro)
        y, x = micro_idx // self.w, micro_idx % self.w
        meso_y, meso_x = y // 4, x // 4
        meso_idx = self.num_micro + (meso_y * (self.w // 4) + meso_x)
        
        edges.append(torch.stack([micro_idx, meso_idx], dim=0)) # Micro -> Meso
        edges.append(torch.stack([meso_idx, micro_idx], dim=0)) # Meso -> Micro
        
        # 3. Meso-to-Macro edges (Connect 4x4 regional blocks to their macro hub)
        meso_local_idx = torch.arange(self.num_meso)
        my, mx = meso_local_idx // (self.w // 4), meso_local_idx % (self.w // 4)
        macro_y, macro_x = my // 4, mx // 4
        macro_idx = self.num_micro + self.num_meso + (macro_y * (self.w // 16) + macro_x)
        global_meso_idx = self.num_micro + meso_local_idx
        
        edges.append(torch.stack([global_meso_idx, macro_idx], dim=0)) # Meso -> Macro
        edges.append(torch.stack([macro_idx, global_meso_idx], dim=0)) # Macro -> Meso
        
        return torch.cat(edges, dim=1)

    def _get_batched_edge_index(self, batch_size, device):
        """Offsets node indices so samples in a batch don't share edges."""
        if batch_size == self.cached_batch_size and self.cached_edge_index is not None:
            return self.cached_edge_index

        edge_indices = []
        for i in range(batch_size):
            offset = i * self.total_nodes
            edge_indices.append(self.base_edge_index + offset)
            
        self.cached_edge_index = torch.cat(edge_indices, dim=1).to(device)
        self.cached_batch_size = batch_size
        return self.cached_edge_index

    def forward(self, x):
        # Handle both [B, 24, 1, 64, 64] and [B, 24, 64, 64]
        if x.dim() == 5: 
            x = x.squeeze(2) 
            
        B, T, H, W = x.shape
        
        # 1. Flatten pixels: Shape -> [B, 4096, 24]
        micro_nodes = x.permute(0, 2, 3, 1).reshape(B, self.num_micro, T)
        
        # 2. Create empty hub nodes for Level 2 and Level 3: Shape -> [B, 272, 24]
        hub_nodes = torch.zeros((B, self.num_meso + self.num_macro, T), dtype=x.dtype, device=x.device)
        
        # 3. Concatenate to form the full graph per batch item -> [B * 4368, 24]
        all_nodes = torch.cat([micro_nodes, hub_nodes], dim=1).view(-1, T)
        
        # 4. Get batched edges
        edge_index = self._get_batched_edge_index(B, x.device)
        
        # 5. Message Passing (Information flows instantly up and down the hierarchy)
        out = self.act1(self.conv1(all_nodes, edge_index))
        out = self.act2(self.conv2(out, edge_index))
        out = self.act3(self.conv3(out, edge_index))
        out = self.conv4(out, edge_index)
        
        # 6. Reshape back to batches: [B, 4368, 24]
        out = out.view(B, self.total_nodes, -1)
        
        # 7. Slice out only the 4096 pixel nodes (we don't output the hubs)
        micro_out = out[:, :self.num_micro, :]
        
        # 8. Reshape back to target 5D image grid pipeline format: [B, 24, 1, 64, 64]
        micro_out = micro_out.view(B, H, W, T).permute(0, 3, 1, 2).unsqueeze(2)
        return micro_out

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epoch, eta_min=1e-6)
        return [optimizer], [scheduler]
    
