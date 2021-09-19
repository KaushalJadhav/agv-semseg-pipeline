from self.models.enet import ENet
from dataloader.cityscapes import CityScapesDataLoader

import torch.nn.functional as F

    
class Tester():
    def __init__(self,model):
        self.model=model
    def final_metrics(self,train_DataLoader, val_DataLoader,num_classes):
     self.model.eval()
    
     train_accuracy = np.zeros((num_classes,), dtype=float)
     train_iou = np.zeros((num_classes,), dtype=float)
    
     val_accuracy = np.zeros((num_classes,), dtype=float)
     val_iou = np.zeros((num_classes,), dtype=float)

     val_results = []
      
     for batch in train_DataLoader:
        
         inputs = batch[0].float().to(device)
         labels = batch[1].float().to(device).long()
        
         outputs = self.model(inputs)

         iou, accu = metrics(outputs, labels, num_classes, True)
        
         train_accuracy += accu
         train_iou += iou
    
     for batch in val_DataLoader:
        
         inputs = batch[0].float().to(device)
         labels = batch[1].float().to(device).long()

         outputs = self.model(inputs)
        
         np_outputs, iou, accu = metrics(outputs, labels, num_classes)
        
         val_accuracy += accu
         val_iou += iou
         val_results.append(np_outputs)
        
     train_accuracy /= len(train_DataLoader)
     val_accuracy /= len(val_DataLoader)
    
     train_iou /= len(train_DataLoader)
     val_iou /= len(val_DataLoader)

     val_results = np.array(val_results)

     return train_accuracy, val_accuracy, train_iou, val_iou, val_results


