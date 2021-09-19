from utils.losses import CrossEntropyLoss
from utils.device import default_device

from models.enet import ENet
from dataloader.cityscapes import CityScapesDataLoader, load_data, get_data_array, get_data

import torch 
import numpy as np 
import torch 

epochs=config.max_epoch
train_batch_size=config.train_batch_size

train_X, train_y, val_X, val_y, test_X = get_data_array(config["train_X_path"], config["train_y_path"], config["val_X_path"], config["val_y_path"], config["test_X_path"])
#Add preprocessing here
train_data, val_data = get_data(train_X, train_y, val_X, val_y)
train_loader, val_loader = load_data(train_data, val_data)




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
        pass


    def validate(self):
        pass



    def finalize(self):
        pass


    
