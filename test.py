import torch
from torch.utils.data import DataLoader, random_split
from model import FocusNet
from torchvision import transforms
from torchvision.utils import save_image
from torch.nn import SmoothL1Loss

from tqdm import tqdm
import time

from load_dataset import Dataset
from CropTransform import ConditionalRandomCrop
from gaussian_noise import GaussianNoise
from poisson_noise import PoissonNoise

from config import dataset_path, batch_size, crop_size, dataset_root

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_tensor, target_tensor in tqdm(test_loader):
            # save_image(input_tensor, "noisy.jpg")
            # break
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            output = model(input_tensor)
            loss = criterion(output, target_tensor)
            total_loss += loss.item() * input_tensor.size(0)
    avg_loss = total_loss / len(test_loader.dataset)
    return avg_loss

if __name__ == '__main__':

    # Преобразования для изображений
    transform = transforms.Compose([
        transforms.ToTensor(),
        PoissonNoise(),
        GaussianNoise(),
        transforms.ToPILImage(),
        ConditionalRandomCrop(crop_size),
        transforms.ToTensor()
    ])

    # Создание тестового датасета и загрузчика данных
    dataset = Dataset(dataset_path, transform)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Функция потерь
    criterion = SmoothL1Loss()

    # Тестирование модели
    device = torch.device("cuda")

    # Создание экземпляра модели
    model = FocusNet()
    model.to(device)

    # Загрузка сохраненных весов
    model.load_state_dict(torch.load('weights/focusnet_weights_01-06-2024_07-56.pth'))

    start = time.time()
    test_loss = test(model, test_loader, criterion, device)
    end = time.time()
    avg_time = (end-start)*1000/len(test_loader.dataset)
    print('Test loss: ', test_loss)
    print('Average CPU time: ', avg_time)

    exit()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        model.load_state_dict(torch.load('weights/focusnet_weights_01-06-2024_07-56.pth'))
        start = time.time()
        test_loss = test(model, test_loader, criterion, device)
        end = time.time()
        avg_time = (end-start)*1000/len(test_loader.dataset)
        print('Average GPU time: ', avg_time)
