from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(
      mean = [0.5, 0.5, 0.5],
      std =  [0.5, 0.5, 0.5],
      ),
])

class CityScapesCutMix(Dataset):

  def __init__(self, image, label, transforms, num_mix=1, beta=1.0, prob=1.0):

    self.data = image
    self.label = label
    self.transform = transforms
    self.num_mix = num_mix
    self.beta = beta
    self.prob = prob

  def __getitem__(self, index):
    image = (self.data[index]).copy()
    label = (self.label[index]).copy()

    for _ in range(self.num_mix):
        r = np.random.rand(1)
        if self.beta <= 0 or r > self.prob:
            continue

        lam = np.random.beta(self.beta, self.beta)
        rand_index = random.choice(range(len(self)))


        image2 = self.data[rand_index]
        label2 = self.label[rand_index]

        bbx1, bby1, bbx2, bby2 = rand_bbox(image.shape, lam)

        image[bby1:bby2, bbx1:bbx2, :] = image2[bby1:bby2, bbx1:bbx2, :]
        label[bby1:bby2, bbx1:bbx2] = label2[bby1:bby2, bbx1:bbx2]


    if self.transform is not None:
       image = self.transform(image)

    return image, label

  def __len__(self):

    return len(self.data)

def get_loader(dataset):
  
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
  train_data = CityScapesCutMix(train_X, train_y, transforms)
  val_data = CityScapesCutMix(val_X, val_y, transforms)
  return train_data, val_data

def load_data(train_data, val_data, config):

  train_loader = get_loader(train_data, config)
  val_loader = get_loader(val_data, config)
  return train_loader, val_loader