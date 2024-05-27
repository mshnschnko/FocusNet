import torch
from torch import nn

if __name__ == '__main__':
    tensor: torch.Tensor = torch.randn(3, 672, 672)
    print(tensor)
    layers = {}
    # exit()
    # conv 1
    layers["conv 1"] = nn.Conv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=2,
        padding=1)
    
    # bneck 1
    layers["bneck 1"] = nn.Conv2d(
        in_channels=16,
        out_channels=16,
        kernel_size=3,
        stride=2,
        padding=1)
    
    # bneck 2
    layers["bneck 2"] = nn.Conv2d(
        in_channels=16,
        out_channels=24,
        kernel_size=3,
        stride=2,
        padding=1)
    
    # bneck 3
    layers["bneck 3"] = nn.Conv2d(
        in_channels=24,
        out_channels=24,
        kernel_size=3,
        stride=1,
        padding=1)
    
    # bneck 4
    layers["bneck 4"] = nn.Conv2d(
        in_channels=24,
        out_channels=40,
        kernel_size=5,
        stride=2,
        padding=2)
    
    # bneck 5
    layers["bneck 5"] = nn.Conv2d(
        in_channels=40,
        out_channels=40,
        kernel_size=5,
        stride=1,
        padding=2)
    
    # bneck 6
    layers["bneck 6"] = nn.Conv2d(
        in_channels=40,
        out_channels=40,
        kernel_size=5,
        stride=1,
        padding=2)
    
    # bneck 7
    layers["bneck 7"] = nn.Conv2d(
        in_channels=40,
        out_channels=48,
        kernel_size=5,
        stride=1,
        padding=2)
    
    # bneck 8
    layers["bneck 8"] = nn.Conv2d(
        in_channels=48,
        out_channels=48,
        kernel_size=5,
        stride=1,
        padding=2)
    
    # bneck 9
    layers["bneck 9"] = nn.Conv2d(
        in_channels=48,
        out_channels=96,
        kernel_size=5,
        stride=2,
        padding=2)
    
    # bneck 10
    layers["bneck 10"] = nn.Conv2d(
        in_channels=96,
        out_channels=96,
        kernel_size=5,
        stride=1,
        padding=2)
    
    # bneck 11
    layers["bneck 11"] = nn.Conv2d(
        in_channels=96,
        out_channels=96,
        kernel_size=5,
        stride=1,
        padding=2)
    
    # conv 2
    layers["conv 2"] = nn.Conv2d(
        in_channels=96,
        out_channels=576,
        kernel_size=1,
        stride=1,
        padding=0)
    
    avg = nn.AdaptiveAvgPool2d(1)
    
    # output_tensor = conv(input_tensor)
    # print(output_tensor.shape)

    for i, layer in layers.items():
        print(f"{i} слой: {tensor.shape}")
        tensor = layer(tensor)

    print(f"avg pooling слой: {tensor.shape}")
    tensor = avg(tensor)

    tensor = tensor.flatten()
    fc1 = nn.Linear(576, 256)
    relu = nn.ReLU()
    fc2 = nn.Linear(256, 1)

    print(f"fc1: {tensor.shape}")
    tensor = fc1(tensor)
    tensor = relu(tensor)

    print(f"fc2: {tensor.shape}")
    tensor = fc2(tensor)

    # regressor = nn.Sequential(
    #         nn.Linear(576, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, 1),
    #     )
    # tensor = regressor(tensor)
    

    print(f"result: {tensor.shape}")