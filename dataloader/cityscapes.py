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

class CityScapesDataLoader():
    def __init__(self, config):
        self.config = config
        self.train_dataloader
        self.valid_dataloader
    
    def get_loader(config, dataset):

      params = {
      'batch_size': config.batch_size,
      'num_workers': config.num_workers,
      'shuffle': True
      }

      dataloader = DataLoader(dataset, **params)
      return dataloader