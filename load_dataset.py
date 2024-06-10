import os
import re
from typing import Callable
from PIL import Image
from torch.utils.data import Dataset as Dataset_
from torchvision import transforms

from config import dataset_path, crop_size, dataset_size

class Dataset(Dataset_):
    def __init__(self, data_dir: str, transform: Callable | None = None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_names = []

        idx = 0

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                # if is_test and idx < stop:
                #     idx += 1
                #     continue
                self.image_names.append(os.path.join(root, file))
            #     idx += 1
            #     if idx == dataset_size:
            #         break
            # if idx == dataset_size:
            #     break
        print(len(self.image_names))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = self.image_names[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            # image.save("clear.jpg")
            image = self.transform(image)

        match = re.search(r".*defocus(-?[0-9]+)\.jp[e]?g", image_path)
        target = int(match.group(1))  # Например, если изображение называется "42.jpg", target будет 42

        return image, target
    