from metric import iou_accu
import numpy as np

# These are utility functions, we need not to initialize any dataloader or model here (ENet or Cityscapes)
# Just pass those in these functions wherever these are called (test.py or train.py) ~mradul

def final_metrics(config, model, train_loader, valid_loader, device):
    model.eval()
    
    val_results = []

    train_accuracy = np.zeros((config.num_classes,), dtype=float)
    train_iou = np.zeros((config.num_classes,), dtype=float)
    
    val_accuracy = np.zeros((config.num_classes,), dtype=float)
    val_iou = np.zeros((config.num_classes,), dtype=float)
      
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
        val_result.append(np_outputs)
        
        val_accuracy += accu
        val_iou += iou
        
    train_accuracy /= len(train_loader)
    val_accuracy /= len(val_loader)
    
    train_iou /= len(train_loader)
    val_iou /= len(val_loader)

    val_results = np.array(val_results)

    return train_accuracy, val_accuracy, train_iou, val_iou, val_results 