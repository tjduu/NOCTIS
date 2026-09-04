from neuralop.models import FNO
import torch.nn as nn
import lightning as L
import torch
from noctis.forecast.loss import PersistenceDefeatingLoss

class FNOForecaster(L.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.save_hyperparameters(config)
        
        self.model = FNO(
            n_modes=(config['modes'], config['modes']), 
            hidden_channels=config['hidden_channels'],
            in_channels=config['in_channels'], 
            out_channels=config['out_channels'],
            n_layers=4 
        )
        self.criterion = PersistenceDefeatingLoss()

    def forward(self, x):
        # 1. Check if input is 5D: [Batch, Time, Channel, H, W]
        if x.dim() == 5:
            x = x.squeeze(2) # Fold Time into Channels -> [B, 24, H, W]
            
        # 2. Run FNO Prediction
        out = self.model(x)  # Output is [B, 24, H, W]
        
        # 3. Un-fold back to 5D so your evaluation script's math works flawlessly
        out = out.unsqueeze(2) # -> [B, 24, 1, H, W]
        
        return out

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        
        pred = self(x) # self(x) now handles the 5D/4D shapes!
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epoch, eta_min=1e-6)
        return [optimizer], [scheduler]