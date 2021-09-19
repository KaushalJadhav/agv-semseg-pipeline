from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms



transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(
      mean = [0.5, 0.5, 0.5],
      std =  [0.5, 0.5, 0.5],
      ),
])

class CityScapes(Dataset):

  def __init__(self, image, label, transforms):

    self.data = image
    self.label = label
    self.transform = transforms

  def __getitem__(self, index):
    image = self.data[index]
    label = self.label[index]

    if self.transform is not None:
        image = self.transform(image)

    return image, label

  def __len__(self):

    return len(self.data)


def get_loader(dataset, config):
  
  """
  Args:
  dataset (Dataset): Custom Dataset class
  Returns:
  DataLoader: Dataloader for training or testing
  """

  params = {
  'batch_size': config.batch_size,
  'num_workers': config.num_workers,
  'shuffle': True
  }

  dataloader = DataLoader(dataset, **params)
  return dataloader


def get_data_array(train_data_path, val_data_path, test_data_path = None):
  train_X = np.load(Train_X_path)
  train_y = np.load(Train_Y_path)
  val_X = np.load(val_X_path)
  val_y = np.load(val_Y_path)
  test_X = None
  if test_X_path:
    test_x = np.load(test_X_path)
  return train_X, train_y, val_X, val_y, test_X

def get_data(train_X, train_y, val_X, val_y):
  train_data = CityScapes(train_X, train_y, transforms)
  val_data = CityScapes(val_X, val_y, transforms)
  return train_data, val_data

def load_data(train_data, val_data, config):

  train_loader = get_loader(train_data, config)
  val_loader = get_loader(val_data, config)
  return train_loader, val_loader
