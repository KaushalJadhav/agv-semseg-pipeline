from dataset import *
import config

epochs=config.max_epoch
train_batch_size=config.train_batch_size
data_path=config.data_path


dataset = Dataset(data_path=data_path)
data_loader = Data_loasder(dataset,num_workers=num_workers,batch_size=train_batch_size,shuffle=shuffle)



trainer(epochs=epochs, model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, checkpoint_path, best_model_path)




