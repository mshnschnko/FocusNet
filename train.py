import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import SmoothL1Loss
from torchvision import transforms

import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from load_dataset import Dataset
from model import FocusNet

from config import batch_size, epoch_num, learning_rate, dataset_path, crop_size

def train(model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
          train_loader: DataLoader, device: torch.device, epoch_num: int):
    model.train()
    loss_values = []

    for epoch in range(epoch_num):
        epoch_loss = 0
        for i, (input_tensor, target_tensor) in enumerate(tqdm(train_loader)):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            
            # Forward pass
            outputs = model(input_tensor)
        
            # Вычисление функции потерь
            loss = criterion(outputs, target_tensor)
        
            # Обратное распространение ошибки и обновление весов
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_values.append(avg_epoch_loss)
        
        # Вывод прогресса обучения
        print(f'Epoch [{epoch+1}/{epoch_num}], Loss: {avg_epoch_loss}')
    
    return loss_values

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FocusNet()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = SmoothL1Loss()

    # Преобразования для изображений
    transform = transforms.Compose([
        # transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor()
    ])

    # Создание датасета и загрузчика данных
    train_dataset = Dataset(dataset_path, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Обучение модели
    loss_values = train(model, optimizer, criterion, train_loader, device, epoch_num)

    # Сохранение модели и весов
    torch.save(model.state_dict(), f'focusnet_weights_{str(datetime.now().strftime("%d-%m-%Y_%H-%M"))}.pth')
    torch.save(model, f'focusnet_model_{str(datetime.now().strftime("%d-%m-%Y_%H-%M"))}.pth')

    # Построение графика потерь
    plt.plot(range(1, epoch_num + 1), loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
