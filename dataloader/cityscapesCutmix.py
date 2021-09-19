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

