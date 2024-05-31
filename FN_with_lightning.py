from torch import nn
import pytorch_lightning as pl
import torch
from torchvision.models.mobilenetv3 import mobilenet_v3_small
import torch.nn.functional as F

from config import learning_rate

class FocusNet(pl.LightningModule):
    def __init__(self):
        super(FocusNet, self).__init__()
        self.model = mobilenet_v3_small(progress=True)
        self.model.classifier = nn.Identity()
        
        self.regressor = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    
    def forward(self, x):
        x = self.model(x)
        x = self.regressor(x)
        return x.view(-1)
        

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.mse_loss(outputs, labels.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.smooth_l1_loss(outputs, labels.float())
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.mse_loss(outputs, labels.float())
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)