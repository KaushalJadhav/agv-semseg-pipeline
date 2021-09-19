from utils.losses import CrossEntropyLoss
from utils.device import default_device

from models.enet import ENet
from dataloader.cityscapes import CityScapesDataLoader

import torch 
import numpy as np 
import torch.

epochs=config.max_epoch
train_batch_size=config.train_batch_size
data_path=config.data_path


dataset = Dataset(data_path=data_path)
data_loader = Data_loasder(dataset,num_workers=num_workers,batch_size=train_batch_size,shuffle=shuffle)



trainer(epochs=epochs, model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, checkpoint_path, best_model_path)



class ENet():
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

        self.valid_dataloader = self.dataloader.valid_dataloader
        self.train_dataloader = self.dataloader.valid_datalaoder

        self.current_epoch = 1

        self.device = default_device()

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

    def train(self):


    def validate(self):



    def finalize(self):


    
