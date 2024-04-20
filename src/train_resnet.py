import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import os
from PIL import Image

class PersonDetectionDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.non_target_dir = os.path.join(root_dir, 'non_target_train')
        self.target_dir = os.path.join(root_dir, 'target_train')

        # List image files in the non_target directory
        self.non_target_images = [f for f in os.listdir(self.non_target_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
        # List image files in the target directory
        self.target_images = [f for f in os.listdir(self.target_dir) if f.endswith('.jpg') or f.endswith('.png')]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.non_target_images) + len(self.target_images)

    def __getitem__(self, idx):
        if idx < len(self.non_target_images):
            img_name = self.non_target_images[idx]
            img_path = os.path.join(self.non_target_dir, img_name)
            label = 0  # Non-target label
        else:
            img_name = self.target_images[idx - len(self.non_target_images)]
            img_path = os.path.join(self.target_dir, img_name)
            label = 1  # Target label

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image, label

class PersonDetectionDatasetValidation(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.non_target_dir = os.path.join(root_dir, 'non_target_dev')
        self.target_dir = os.path.join(root_dir, 'target_dev')

        self.non_target_images = [f for f in os.listdir(self.non_target_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.target_images = [f for f in os.listdir(self.target_dir) if f.endswith('.jpg') or f.endswith('.png')]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.non_target_images) + len(self.target_images)

    def __getitem__(self, idx):
        if idx < len(self.non_target_images):
            img_name = self.non_target_images[idx]
            img_path = os.path.join(self.non_target_dir, img_name)
            label = 0  # Non-target label
        else:
            img_name = self.target_images[idx - len(self.non_target_images)]
            img_path = os.path.join(self.target_dir, img_name)
            label = 1  # Target label

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image, label


data_dir = '/home/david/Documents/CodingFiles/GitWorkspace/sur/data/SUR_projekt2023-2024'

# Create custom dataset
train_dataset = PersonDetectionDataset(data_dir)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create custom dataset for validation
val_dataset = PersonDetectionDatasetValidation(data_dir)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ResNet18 model
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=False)  # Initialize with random weights
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)  # Output 1 value (person or no person)

    def forward(self, x):
        return self.resnet(x)

model = ResNet18()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 5

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()

    accuracy = correct / total
    print(f'Validation Accuracy after Epoch [{epoch+1}/{num_epochs}]: {accuracy:.4f}')

# Evaluate the final model on validation dataset
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.round(torch.sigmoid(outputs))
        total += labels.size(0)
        correct += (predicted == labels.unsqueeze(1)).sum().item()

final_validation_accuracy = correct / total
print(f'Final Validation Accuracy: {final_validation_accuracy:.4f}')

torch.save(model.state_dict(), 'person_detector_resnet18.pth')
