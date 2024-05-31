import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.nn import SmoothL1Loss
from torchvision import transforms

import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import time

from load_dataset import Dataset
from model import FocusNet
from test import test

from config import batch_size, epoch_num, learning_rate, dataset_path, crop_size

# print(torch.cuda.is_available())
# exit()

def train(model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
          train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
          epoch_num: int, scheduler: _LRScheduler | None = None):
    model.train()
    train_loss_values = []
    val_loss_values = []
    current_lr = learning_rate

    for epoch in range(epoch_num):
        model.train()
        epoch_train_loss = 0
        for i, (input_tensor, target_tensor) in enumerate(tqdm(train_loader)):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            
            outputs = model(input_tensor)
        
            loss = criterion(outputs, target_tensor)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            epoch_train_loss += loss.item()
        
        avg_epoch_train_loss = epoch_train_loss / len(train_loader)
        train_loss_values.append(avg_epoch_train_loss)

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for i, (input_tensor, target_tensor) in enumerate(val_loader):
                input_tensor = input_tensor.to(device)
                target_tensor = target_tensor.to(device)
                
                # Forward pass
                outputs = model(input_tensor)
                
                # Вычисление функции потерь
                loss = criterion(outputs, target_tensor)
                
                epoch_val_loss += loss.item()
        
        avg_epoch_val_loss = epoch_val_loss / len(val_loader)
        val_loss_values.append(avg_epoch_val_loss)

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_epoch_val_loss)
            else:
                scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        # Вывод прогресса обучения
        print(f'Epoch [{epoch+1}/{epoch_num}], Train Loss: {avg_epoch_train_loss}, Val Loss: {avg_epoch_val_loss}, Current LR: {current_lr}')
    
    return train_loss_values, val_loss_values

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FocusNet()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=1)

    criterion = SmoothL1Loss()

    # Преобразования для изображений
    transform = transforms.Compose([
        transforms.Lambda(
            lambda img: transforms.RandomCrop((crop_size, crop_size))(img) if img.size[0] > crop_size and img.size[1] > crop_size else img
        ),
        transforms.ToTensor()
    ])

    # Создание датасета и загрузчика данных
    dataset = Dataset(dataset_path, transform)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Обучение модели
    train_loss_values, val_loss_values = train(model, optimizer, criterion, train_loader, val_loader, device, epoch_num)

    # тестирование на незнакомых данных
    start = time.time()
    test_loss = test(model, test_loader, criterion, device)
    end = time.time()
    avg_time = (end-start)*1000/len(test_loader.dataset)
    print('Test loss: ', test_loss)
    print('Average time: ', avg_time)

    # Сохранение модели и весов
    torch.save(model.state_dict(), f'weights/focusnet_weights_{str(datetime.now().strftime("%d-%m-%Y_%H-%M"))}.pth')
    torch.save(model, f'models/focusnet_model_{str(datetime.now().strftime("%d-%m-%Y_%H-%M"))}.pth')

    # Построение графика потерь
    plt.plot(range(1, epoch_num + 1), train_loss_values, label='Train Loss')
    plt.plot(range(1, epoch_num + 1), val_loss_values, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'losses_{str(datetime.now().strftime("%d-%m-%Y_%H-%M"))}.png')
    plt.show()
