from models.enet import ENet
from dataloader.cityscapes import CityScapesDataLoader

import torch.nn.functional as F

def evaluate(model, val_loader, criterion, num_classes, idx_val):
    
    model.eval()
    val_loss = 0.0    
    
    for batch in val_loader:
        
        inputs = batch[0].float().to(device)
        labels = batch[1].float().to(device).long()

        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        val_loss += loss.item()
        
        #wandb.log({'Batch Loss/Val': loss, 'idx_val': idx_val})
        
        idx_val += 1
        
    return val_loss / len(val_loader)
        

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, num_classes, checkpoint_path, best_model_path):

    idx_train = 1
    idx_val = 1

    val_loss_min = 9.9

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0

        for batch in train_loader:

            inputs = batch[0].float().to(device)
            labels = batch[1].float().to(device).long()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # wandb.log({'Batch Loss/Train': loss, 'idx_train': idx_train})
            
            idx_train += 1
            
        scheduler.step()

        val_loss = evaluate(model, val_loader, criterion, num_classes, idx_val)
        idx_val += len(val_DataLoader)
        train_loss = train_loss / len(train_loader)

        # wandb.log({'Loss/Train': train_loss,
        #            'Loss/Val': val_loss,
        #            'Epoch': epoch+1
        #             })
        
        
        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'val_loss_min': val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
        ## TODO: save the model if validation loss has decreased
        if val_loss <= val_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_loss_min,val_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            val_loss_min = val_loss
            
    # return trained model
    return model
                                                  


class Trainer():
    def __init__(self):
