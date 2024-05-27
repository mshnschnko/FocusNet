import torch
from torch.utils.data import DataLoader
from load_dataset import Dataset
from model import FocusNet
from torchvision import transforms
from torch.nn import SmoothL1Loss

from tqdm import tqdm

from config import dataset_path, batch_size

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
    print(f'Test Loss: {avg_loss}')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Преобразования для изображений
    transform = transforms.Compose([
        # transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor()
    ])

    # Создание тестового датасета и загрузчика данных
    test_dataset = Dataset(dataset_path, transform, True)  # Замените test_dataset_path на путь к вашему тестовому датасету
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Создание экземпляра модели
    model = FocusNet()
    model.to(device)

    # Загрузка сохраненных весов
    model.load_state_dict(torch.load('focusnet_weights_27-05-2024_10-43.pth'))  # Замените 'focusnet_weights.pth' на путь к вашим сохраненным весам

    # Функция потерь
    criterion = SmoothL1Loss()

    # Тестирование модели
    test(model, test_loader, criterion, device)
