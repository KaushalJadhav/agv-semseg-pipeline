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

          print("Total number of train images loaded: ", self.train_X.shape[0])

          self.valid_X = np.load(self.config.valid_X_path)
          self.valid_y = np.load(self.config.valid_y_path)

          print("Total number of valid images loaded: ", self.valid_X.shape[0])

          # encoding labels from 35 to 19 classes 
          print("Encoding labels to 19 classes...")
          self.train_y, self.valid_y = encode_data(self.train_y, self.valid_y)

          print("Different classes present in the label: ", np.unique(self.train_y))

          self.train_set = CityScapes(self.train_X,
                          self.train_y, 
                          transforms=self.transform)
          self.valid_set = CityScapes(self.valid_X,
                          self.valid_y,
                          transforms=self.transform)

          self.train_loader = DataLoader(self.train_set, batch_size=self.config.train_batch_size, shuffle=True)
          self.valid_loader = DataLoader(self.valid_set, batch_size=self.config.valid_batch_size, shuffle=False)

          print("Size of train_loader: ", len(self.train_loader))
          print("Size of valid_loader: ", len(self.valid_loader))



        elif self.config.mode == 'test':

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
