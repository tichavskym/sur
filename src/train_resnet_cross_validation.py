import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from sklearn.model_selection import KFold

import random

class RandomAugmentation:
    def __init__(self):
        self.transforms = [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([transforms.RandomRotation(degrees=15)], p=0.5),
            transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2)], p=0.3),
            transforms.RandomApply([transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))], p=0.3)
        ]

    def __call__(self, img):
        if random.random() < 0.8:
            transform = random.choice(self.transforms)
            img = transform(img)

        return img

class PersonDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.non_target_dir = os.path.join(root_dir, 'non_target_train')
        self.target_dir = os.path.join(root_dir, 'target_train')

        self.non_target_images = [os.path.join(self.non_target_dir, f) for f in os.listdir(self.non_target_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.target_images = [os.path.join(self.target_dir, f) for f in os.listdir(self.target_dir) if f.endswith('.jpg') or f.endswith('.png')]

        self.transform = transform

    def __len__(self):
        return len(self.non_target_images) + len(self.target_images)

    def __getitem__(self, idx):
        if idx < len(self.non_target_images):
            img_path = self.non_target_images[idx]
            label = 0  # Non-target label
        else:
            img_path = self.target_images[idx - len(self.non_target_images)]
            label = 1  # Target label

        image = Image.open(img_path).convert('RGB')
        if self.transform:
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


# ResNet18 model
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.resnet(x)

def train_and_save_model(data_dir, num_epochs=20, batch_size=32, learning_rate=0.001, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    best_accuracy = 0.0
    best_model_state_dict = None

    transform = transforms.Compose([        
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        RandomAugmentation(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create custom dataset
    dataset = PersonDetectionDataset(data_dir, transform=transform)

    fold_idx = 0
    for train_idx, val_idx in kf.split(dataset):
        fold_idx += 1
        print(f'Fold {fold_idx}')

        # Create DataLoader for training and validation sets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = ResNet18()
        criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_subset)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Evaluation on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                predicted = torch.round(torch.sigmoid(outputs))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Validation Accuracy: {accuracy:.4f}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state_dict = model.state_dict()

    # Create custom dataset for validation
    val_dataset = PersonDetectionDatasetValidation(data_dir)
    # Create DataLoader for validation
    val_loader_complete = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Evaluation on validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader_complete:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Save the best model
    if best_model_state_dict is not None:
        torch.save(best_model_state_dict, 'person_detector_resnet18_cross.pth')
        print('Best model saved.')

if __name__ == '__main__':
    data_dir = '/home/david/Documents/CodingFiles/GitWorkspace/sur/data/SUR_projekt2023-2024'    
    train_and_save_model(data_dir)
