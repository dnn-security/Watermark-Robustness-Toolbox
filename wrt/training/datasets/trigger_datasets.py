import os

import mlconfig
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive


class Trigger(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.items = os.listdir(self.root)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.items[idx]
        image = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image


class AdiTrigger(Trigger):
    url = "https://www.dropbox.com/s/z11ds7jvewkgv18/adi.zip?dl=1"
    filename = "adi.zip"
    folder_name = "adi"

    def __init__(self, root: str, download=True, transform=None):
        """ Trigger images for the Adi watermark.
        Source: https://github.com/adiyoss/WatermarkNN
        """
        self.root = os.path.join(os.path.expanduser(root), self.folder_name)

        if download:
            self.download()
        super().__init__(os.path.abspath(self.root), transform)

    def _check_integrity(self):
        return os.path.isdir(self.root)

    def download(self):
        if self._check_integrity():
            print("Adi trigger set was already downloaded.")
            return
        os.makedirs(self.root, exist_ok=True)
        download_and_extract_archive(self.url, self.root, filename=self.filename, remove_finished=True)


@mlconfig.register
class AdiTriggerDataLoader(data.DataLoader):
    def __init__(self, root, image_size, batch_size=32, shuffle=False, num_workers=0,  **kwargs):

        transform = transforms.Compose([
            transforms.Resize(image_size + 32, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

        dataset = AdiTrigger(root=root, transform=transform)
        super(AdiTriggerDataLoader, self).__init__(dataset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers,
                                                   **kwargs)
