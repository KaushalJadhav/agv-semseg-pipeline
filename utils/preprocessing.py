from utils.scripts import class_labels 

def encode(mask):
    res = np.zeros_like(mask)
    for label in class_labels:
        res[mask == label.id] = label.trainId
    return res

def encode_data(train_y, val_y):
    train_y_encoded = np.array([encode(mask) for mask in train_y])
    val_y_encoded = np.array([encode(mask) for mask in val_y])