import torch
from torch.utils.data import DataLoader, TensorDataset

from ovacadx.models import MILModel


# Create a dummy dataset
dummy_data = torch.randn((100, 1, 28, 28))
dummy_labels = torch.randint(0, 10, (100,))

dataset = TensorDataset(dummy_data, dummy_labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Test initialization
model = MILModel(num_classes=10)

# Test forward pass
for images, labels in dataloader:
    outputs = model(images)
    print(outputs.shape)  # Expected output: torch.Size([2, 10])

# Test forward pass without head
for images, labels in dataloader:
    outputs = model(images, no_head=True)
    print(outputs.shape)  # Expected output: torch.Size([2, 16])

# Test forward pass with different MIL modes
model = MILModel(num_classes=10, mil_mode="mean")
for images, labels in dataloader:
    outputs = model(images)
    print(outputs.shape)  # Expected output: torch.Size([2, 10])

model = MILModel(num_classes=10, mil_mode="max")
for images, labels in dataloader:
    outputs = model(images)
    print(outputs.shape)  # Expected output: torch.Size([2, 10])

model = MILModel(num_classes=10, mil_mode="att")
for images, labels in dataloader:
    outputs = model(images)
    print(outputs.shape)  # Expected output: torch.Size([2, 10])

model = MILModel(num_classes=10, mil_mode="att_trans")
for images, labels in dataloader:
    outputs = model(images)
    print(outputs.shape)  # Expected output: torch.Size([2, 10])

model = MILModel(num_classes=10, mil_mode="att_trans_pyramid")
for images, labels in dataloader:
    outputs = model(images)
    print(outputs.shape)  # Expected output: torch.Size([2, 10])
