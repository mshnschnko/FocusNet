import pytorch_lightning as pl

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from load_dataset import Dataset
from FN_with_lightning import FocusNet
from CropTransform import ConditionalRandomCrop

from config import batch_size, epoch_num, learning_rate, dataset_path, crop_size


if __name__ == '__main__':
    # Создание данных для примера
    # Преобразования для изображений
    transform = transforms.Compose([
        ConditionalRandomCrop(crop_size),
        transforms.ToTensor()
    ])

    # Создание датасета и загрузчика данных
    dataset = Dataset(dataset_path, transform)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=3, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Callback для раннего прекращения
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',  # мониторинг потерь валидации
        patience=5,          # количество эпох для терпения
        verbose=True,        # вывод сообщений
        mode='min'           # минимизация потерь
    )

    # Обучение модели
    model = FocusNet()
    trainer = pl.Trainer(max_epochs=epoch_num, callbacks=[early_stopping_callback])
    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)