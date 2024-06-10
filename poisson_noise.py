import torch
import numpy as np

class PoissonNoise:
    def __init__(self, mean = 30) -> None:
        self.mean = mean

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # scaled_img = img * 255.0  # Преобразуем в диапазон [0, 255]
        # noisy_img = torch.poisson(scaled_img).float() / 255.0  # Применяем шум и возвращаем к диапазону [0, 1]
        noise = torch.Tensor(np.random.poisson(lam = self.mean, size=img.shape)).float() / 255.0
        return img + noise
