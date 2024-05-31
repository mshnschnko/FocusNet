import torch
from torch.utils.data import DataLoader
from load_dataset import Dataset
from model import FocusNet
from torchvision import transforms
from torch.nn import SmoothL1Loss

from tqdm import tqdm
import time

from config import dataset_path, batch_size, crop_size, dataset_root

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_tensor, target_tensor in tqdm(test_loader):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            output = model(input_tensor)
            loss = criterion(output, target_tensor)
            total_loss += loss.item() * input_tensor.size(0)
    avg_loss = total_loss / len(test_loader.dataset)
    return avg_loss

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # Преобразования для изображений
    transform = transforms.Compose([
        transforms.Lambda(
            lambda img: transforms.RandomCrop((crop_size, crop_size))(img) if img.size[0] > crop_size and img.size[1] > crop_size else img
        ),
        transforms.ToTensor()
    ])

    # Создание тестового датасета и загрузчика данных
    test_dataset = Dataset(dataset_path, transform)
    exit()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Создание экземпляра модели
    model = FocusNet()
    model.to(device)

    # Загрузка сохраненных весов
    model.load_state_dict(torch.load('focusnet_weights_29-05-2024_00-47.pth'))

    # Функция потерь
    criterion = SmoothL1Loss()

    # Тестирование модели
    test(model, test_loader, criterion, device)
