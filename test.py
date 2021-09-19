
from dataset import *
import config

class test:
  def __init__(self, model, config):
    self.epochs=config.max_epoch
    self.train_batch_size=config.train_batch_size
    self.data_path=config.data_path


    self.dataset = Dataset(data_path=data_path)
    self.data_loader = Data_loader(dataset,num_workers=num_workers,batch_size=train_batch_size,shuffle=shuffle)

  test(epochs=self.epochs, model=model, train_loader=train_loader, val_loader=val_loader, criterion=self.criterion, optimizer=self.optimizer, scheduler=self.scheduler, checkpoint_path, best_model_path)
