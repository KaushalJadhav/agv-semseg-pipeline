from utils.metric import iou_accu
import numpy as np

# These are utility functions, we need not to initialize any dataloader or model here (ENet or Cityscapes)
# Just pass those in these functions wherever these are called (test.py or train.py) ~mradul

def final_metrics(config, model, train_loader, valid_loader, device):
    model.eval()
    
    valid_results = []

    train_accuracy = np.zeros((config.num_classes,), dtype=float)
    train_iou = np.zeros((config.num_classes,), dtype=float)
    
    valid_accuracy = np.zeros((config.num_classes,), dtype=float)
    valid_iou = np.zeros((config.num_classes,), dtype=float)
      
    for batch in train_loader:
        
        inputs = batch[0].float().to(device)
        labels = batch[1].float().to(device).long()
        
        outputs = model(inputs)

        np_outputs, iou, accu = iou_accu(config, outputs, labels)
        
        train_accuracy += accu
        train_iou += iou
    
    for batch in valid_loader:
        
        inputs = batch[0].float().to(device)
        labels = batch[1].float().to(device).long()

        outputs = model(inputs)
        
        np_outputs, iou, accu = iou_accu(config, outputs, labels)
        valid_results.append(np_outputs)
        
        valid_accuracy += accu
        valid_iou += iou
        
    train_accuracy /= len(train_loader)
    valid_accuracy /= len(valid_loader)
    
    train_iou /= len(train_loader)
    valid_iou /= len(valid_loader)

    valid_results = np.array(valid_results)

    return train_accuracy, valid_accuracy, train_iou, valid_iou, valid_results 