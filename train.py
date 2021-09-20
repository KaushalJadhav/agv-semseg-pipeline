from dataloader.cityscapes import CityScapesDataLoader
from dataloader.cityscapes_cutmix import CityScapesCutmixDataLoader
# Call dataloader class from dataloader package and initialize them wherever needed ~mradul2

import torch.optim as optim
import torch.nn as nn          #We wll make this modular by adding these either to config

from models.enet import ENet
import torch 
import numpy as np 


#wandb 

class Train():
    def __init__(self, config):
        self.config = config
        self.model = ENet(self.config)
        self.dataloader = CityScapesDataLoader(self.config)
        self.loss = CrossEntropyLoss(self.config)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.learning_rate,
                                          weight_decay=self.config.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=self.config.gamma)

        self.valid_dataloader = self.dataloader.valid_loader
        self.train_dataloader = self.dataloader.train_loader

        self.current_epoch = 1

        self.device = default_device()
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

    def forward(self):

        trainer(epochs=self.epochs, model=self.model, train_loader=self.train_dataloader, val_loader=self.valid_dataloader, criterion=self.loss, optimizer=self.optimizer, scheduler=self.scheduler, checkpoint_path, best_model_path)
        #Here still we have to fix the path to where we log the data


    
