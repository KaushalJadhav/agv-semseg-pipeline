import torch.nn.functional as F
import numpy as np
import torch
from utils.augmentation import rand_bbox

def train_one_epoch(config, model, train_dataloader, optimizer, loss_fun):
    model.train()
    train_loss = 0.0
    
    for batch in train_dataloader:
        inputs = batch[0].float().to(device)
        labels = batch[1].float().to(device).long()

        r = np.random.rand(1)
        if self.config.cutmix and r < config.cutmix_prob:
            lam = np.random.beta(config.beta_cutmix, config.beta_cutmix)
            rand_index = torch.randperm(inputs.size()[0]).cuda()

            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)

            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            labels[:, bbx1:bbx2, bby1:bby2] = labels[rand_index, bbx1:bbx2, bby1:bby2]

        outputs = model(inputs)
        loss = loss_fun(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    return train_loss / len(train_dataloader)
                                   

def validate(config, model, valid_loader, loss_fun):
    
    model.eval()
    valid_loss = 0.0    
    
    for batch in valid_loader:
        
        inputs = batch[0].float().to(device)
        labels = batch[1].float().to(device).long()

        outputs = model(inputs)
        
        loss = loss_fun(outputs, labels)
        
        val_loss += loss.item()
        
    return val_loss / len(val_loader)
        
        
