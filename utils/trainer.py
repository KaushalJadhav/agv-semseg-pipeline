import torch.nn.functional as F

def train_one_epoch(config, model, train_dataloader, optimizer, loss_fun):
    model.train()
    train_loss = 0.0
    
    for batch in train_dataloader:
        inputs = batch[0].float().to(device)
        labels = batch[1].float().to(device).long()

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
        
        
