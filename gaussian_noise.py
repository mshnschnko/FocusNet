import torch

class GaussianNoise:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noise = torch.randn_like(img) * self.std + self.mean
        return img + noise