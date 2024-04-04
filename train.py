import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import SmoothL1Loss

import config
from model import FocusNet

def train(model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, train_dataset,
          epoch_num: int = 50, batch_size: int = 1, learning_rate: float = 0.001):
    for epoch in range(epoch_num):
        for i, (input_tensor, target_tensor) in enumerate(train_dataset):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            
            # Forward pass
        # Forward pass
        outputs = model(input_tensor)
        
        # Вычисление функции потерь
        loss = criterion(outputs, target_tensor)
        
        # Обратное распространение ошибки и обновление весов
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Вывод прогресса обучения
        print(f'Epoch [{epoch+1}/{epoch_num}], Loss: {loss.item()}')

    
if __name__ == '__main__':
    model = FocusNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    criterion = SmoothL1Loss()

    train_dataset = torch.randn(10, 3, 672, 672)
    train_dataset = Dataset(train_dataset)
    # target_dataset = torch.randn(10, 3, 672, 672)
    # train_dataloader = DataLoader(, batch_size=config.batch_size, shuffle=True)

    # train(model, optimizer, criterion, train_dataset, config.epoch_num, config.batch_size, config.learning_rate)