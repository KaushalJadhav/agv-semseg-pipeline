import torch
import numpy as np

# These are utility functions, we need not to initialize any dataloader or model here (ENet or Cityscapes)
# Just pass those in these functions wherever these are called (test.py or train.py) ~mradul

def iou_accu(config, outputs, labels):
    accu = np.zeros((config.num_classes,), dtype=float)
    iou = np.zeros((config.num_classes,), dtype=float)

    output_cvt = torch.argmax(outputs, dim=1)

    np_outputs = output_cvt.cpu().detach().numpy()
    np_labels = labels.cpu().detach().numpy()

    np_outputs[np_labels == config.ignore_class] = config.ignore_class

    for x in range(config.num_classes):
        output_mask = (np_outputs == x)
        label_mask = (np_labels == x)
    
        intersection = (output_mask & label_mask).sum((1, 2))
        union = (output_mask | label_mask).sum((1, 2))
        total = label_mask.sum((1, 2))

        iou[x] = ((intersection + config.SMOOTH) / (union + config.SMOOTH)).mean()
        accu[x] = ((intersection + config.SMOOTH) / (total + config.SMOOTH)).mean()
    
    return iou, accu