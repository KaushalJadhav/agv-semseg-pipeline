from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np

from utils.preprocessing import encode_data


# Dataset and Dataloader have to be in seperate classes 
# We will call high level dataloader class from test/train.py which will wrap dataset class ~mradul

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


# Now by simply calling this class in test/train.py all loaders will be initialized and can be acessed too 
# No need to call any forward or initialise method with this class 

class CityScapesDataLoader:
    def __init__(self, config):
        self.config = config

        self.transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(
              mean = [0.5, 0.5, 0.5],
              std =  [0.5, 0.5, 0.5],
              )
        ])

        if self.config.mode == 'train':

          self.train_X = np.load(self.config.train_X_path)
          self.train_y = np.load(self.config.train_y_path)

          self.valid_X = np.load(self.valid_X_path)
          self.valid_y = np.load(self.valid_y_path)

          # encoding labels from 35 to 19 classes 
          encode_data(self.train_y, self.valid_y)

          self.train_set = CityScapes(self.train_X,
                          self.train_y, 
                          transforms=self.transform)
          self.valid_set = CityScapes(self.val_X,
                          self.val_y,
                          transforms=self.transform)

          self.train_loader = DataLoader(self.train_set, batch_size=self.config.train_batch_size, shuffle=True)
          self.valid_loader = DataLoader(self.valid_set, batch_size=self.config.valid_batch_size, shuffle=False)

        if self.config.mode == 'test':

          self.test_X = np.load(self.config.test_X_path)
          self.test_y = np.load(self.config.test_y_path)

          self.test_set = CityScapes(self.test_X,
                          self.test_y, 
                          transforms=self.transform)          

          self.test_loader = DataLoader(self.test_set, batch_size=self.config.test_batch_size, shuffle=False)

        else: 
          print("Invalid mode provided!")


      def finalize():
        print("Dataloaders are finalized..")


        

# our dataloader initialization is different, this won't work though functional modularity could be used
# for get_data_array, get_data, load_data ~mradul2

'''
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
'''