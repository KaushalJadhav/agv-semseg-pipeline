import torch 
import numpy as np 

from dataloader.cityscapes import CityScapesDataLoader

import torch.optim as optim

from models.enet import ENet

from utils.device import default_device
from utils.trainer import train_one_epoch, validate
from utils.tester import final_metrics
from utils.wandb_utils import init_wandb, wandb_log, wandb_save_summary, save_model_wandb
from utils.saving import save_ckp, load_ckp, make_checkpoint_dict
from utils.losses import CrossEntropyLoss

class Train():
    def __init__(self, config):
        self.config = config

        print("Loading ENet Model...")
        self.model = ENet(self.config)

        print("Loading Dataloaders...")
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
        self.min_valid_loss = np.inf

        self.device = default_device()
        print("Default device: ", self.device)

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

        if self.config.wandb:
            # Initialise wandB
            print("Initializing WandB...")
            init_wandb(self.model, self.config)

    def forward(self):
        # So below code calls training loop and validation loop for each epach and:
        # log losses on wandb, save models, step the scheduler 

        print("Training started...")
        # Training loop called
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            # Call training loop for one epoch 
            train_loss = train_one_epoch(self.config, 
                                         self.model,
                                         self.train_dataloader,
                                         self.optimizer,
                                         self.loss.forward,
                                         self.device)
            
            # Validating on the validation set
            valid_loss = validate(self.config,
                                  self.model, 
                                  self.valid_dataloader,
                                  self.loss.forward,
                                  self.device)

            print("Epoch: ", self.current_epoch)
            print("-- train_loss: ", train_loss)
            print("-- valid_loss: ", valid_loss)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            # stepping the scheduler object
            self.scheduler.step()

            # Saving models:
            checkpoint = make_checkpoint_dict(self.model, self.optimizer, self.current_epoch, valid_loss)
            save_ckp(checkpoint, False, self.config.checkpoint_path, self.config.best_model_path)

            if valid_loss <= self.min_valid_loss:
                # save checkpoint as best model
                save_ckp(checkpoint, True, self.config.checkpoint_path, self.config.best_model_path)
                self.min_valid_loss = valid_loss

            if self.config.wandb:
                # Log losses on wandb 
                wandb_log(train_loss, valid_loss, self.current_epoch)

        print("Training loop ends")

    def summarize(self):
        """
            Summarize the training process
            will save the final summaries on wanb, save final models on wanb
            log obtained iou, acuracy and some result images on wanb and finally finish the run
        """
        print("Experiment summary...")

        train_accuracy, valid_accuracy, train_iou, valid_iou, valid_results = final_metrics(self.config,
                                                                         self.model,
                                                                         self.train_dataloader,
                                                                         self.valid_dataloader,
                                                                         self.device)

        
        print("Mean Train Accuracy: ", train_accuracy.mean())
        print("Classwise Train Accuracy: ", train_accuracy)
        print("Mean Valid Accuracy: ", valid_accuracy.mean())
        print("Classwise Valid Accuracy: ", valid_accuracy)
        print("Mean Train IoU: ", train_iou.mean())
        print("Classwise Train IoU: ", train_iou)
        print("Mean Valid IoU: ", valid_iou.mean())
        print("Classwise Valid IoU: ", valid_iou)


        if self.config.wandb: 
            print("Saving Experiment summary on WandB")
            save_model_wandb(self.config.checkpoint_path)
            save_model_wandb(self.config.best_model_path)

            wandb_save_summary(valid_accuracy,
                               valid_iou,
                               train_accuracy,
                               train_iou,
                               valid_results,
                               self.dataloader.valid_X,
                               self.dataloader.valid_y)

        print("Experminent is Finished...")