SMOOTH = 1e-6

def metrics(outputs, labels, num_classes, train = False):

    accu = np.zeros((num_classes,), dtype=float)
    iou = np.zeros((num_classes,), dtype=float)

    output_cvt = torch.argmax(outputs, dim=1)

    np_outputs = output_cvt.cpu().detach().numpy()
    np_labels = labels.cpu().detach().numpy()

    np_outputs[np_labels == 19] = 19

    for x in range(num_classes):
        output_mask = (np_outputs == x)
        label_mask = (np_labels == x)
    
        intersection = (output_mask & label_mask).sum((1, 2))
        union = (output_mask | label_mask).sum((1, 2))
        total = label_mask.sum((1, 2))

        iou[x] = ((intersection + SMOOTH) / (union + SMOOTH)).mean()
        accu[x] = ((intersection + SMOOTH) / (total + SMOOTH)).mean()
    
    if train:
        return iou, accu
    else: 
        return np_outputs.squeeze(), iou, accu