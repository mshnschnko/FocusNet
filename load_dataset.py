import os
import re
from typing import Callable
from PIL import Image
from torch.utils.data import Dataset as Dataset_
from torchvision import transforms

from config import dataset_path, crop_size

class Dataset(Dataset_):
    def __init__(self, data_dir: str, transform: Callable | None = None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_names = []

        idx = 0
        stop = 10000

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                # if is_test and idx < stop:
                #     idx += 1
                #     continue
                self.image_names.append(os.path.join(root, file))
                idx += 1
                if idx == stop:
                    break
            if idx == stop:
                break

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = self.image_names[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        match = re.search(r".*defocus(-?[0-9]+)\.jp[e]?g", image_path)
        target = int(match.group(1))  # Например, если изображение называется "42.jpg", target будет 42

        return image, target
    
if __name__ == "__main__":
    data_dir = dataset_path
    transform = transforms.Compose([
        transforms.Lambda(
            lambda img: transforms.Resize((crop_size, crop_size)) if img.size[0] > crop_size and img.size[1] > crop_size else img
        ),
        transforms.ToTensor()
    ])
    dataset = Dataset(data_dir, transform)
    print(len(dataset))
    im, tg, imp = dataset[1]
    print(imp, tg)