import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


CHECKPOINT_FILE_NAME = 'model_checkpoint.pt'

# The function for parsing the user arguments.
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train the ResNet-18 model')    
    parser.add_argument('--dataset', type=str, required=True, help='Path to the directory containing the complete dataset')
    return parser.parse_args()

# On-the-fly Augmentation class.
#
# The class provides a list of possible augmentation transformations. The 
# transformation is applied with probability 0.8. With that, the chosen 
# transformation is randomly selected too.
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
        # Apply the randomly selected transformation on the original image with
        # the selected probability.
        if random.random() < 0.8:
            transform = random.choice(self.transforms)
            img = transform(img)

        return img

# The class for providing the dataset for the training part and validation part during the training.
class TrainingValidationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.non_target_dir = os.path.join(root_dir, 'non_target_train')
        self.target_dir = os.path.join(root_dir, 'target_train')
        self.non_target_images = [os.path.join(self.non_target_dir, f) for f in os.listdir(self.non_target_dir) if f.endswith('.png')]
        self.target_images = [os.path.join(self.target_dir, f) for f in os.listdir(self.target_dir) if f.endswith('.png')]

        # The normalization values are taken from the following source:
        # Source: https://pytorch.org/vision/0.15/transforms.html#transforms-scriptability
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            RandomAugmentation(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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

# The class for providing the dataset for the testing part.
class TestingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.non_target_dir = os.path.join(root_dir, 'non_target_dev')
        self.target_dir = os.path.join(root_dir, 'target_dev')
        self.non_target_images = [f for f in os.listdir(self.non_target_dir) if f.endswith('.png')]        
        self.target_images = [f for f in os.listdir(self.target_dir) if f.endswith('.png')]

        # The normalization values are taken from the following source:
        # Source: https://pytorch.org/vision/0.15/transforms.html#transforms-scriptability
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

# The EarlyStopping class implementation is based on the code from the following source:
# Source: https://github.com/Bjarten/early-stopping-pytorch/blob/f1a4cad7ebe762c1e3ca9e74c0845a555616952b/pytorchtools.py
# Author: Bjarte Mehus Sunde  (https://github.com/Bjarten)
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, path='checkpoint.pt'):
        self.patience = patience  # How long to wait after last time validation loss improved.
        self.delta = delta  # Minimum change in monitored quantity to qualify as an improvement.        
        self.path = path  #  Path for the checkpoint to be saved to.
        self.counter = 0  # Counter for consecutive epochs without improvement.
        self.best_score = None  # Best validation score.
        self.early_stop = False  # Flag to indicate whether to stop training.
        self.verbose = verbose

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'Early Stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)


# ResNet18 model definition.
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # The weights parameter is by default none, so the non-pre-trained 
        # model is used (only the model architecture).
        self.resnet = models.resnet18()
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.resnet(x)

# Function for training the model, evaluation and final testing.
def train_and_save_model(data_dir, num_epochs=20, batch_size=32, learning_rate=0.0001, num_folds=5):    
    # Create custom training and validation dataset.
    dataset = TrainingValidationDataset(data_dir)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ResNet18()
    # Binary cross-entropy loss.
    criterion = nn.BCEWithLogitsLoss()
    # For the optimizer, the L2 regularization is used.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    early_stopping = EarlyStopping(patience=5, verbose=True, path=CHECKPOINT_FILE_NAME)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    #################
    # Training loop #
    #################
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

        epoch_loss = running_loss / len(dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Evaluation on validation set.
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                predicted = torch.round(torch.sigmoid(outputs))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Validation Accuracy: {accuracy:.4f}')

        val_loss /= len(val_loader)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Create custom dataset for testing.
    test_dataset = TestingDataset(data_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    current_directory = os.getcwd()
    model_path = current_directory + '/' + CHECKPOINT_FILE_NAME

    model = ResNet18()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluation on testing dataset.
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    args = parse_arguments()
    data_dir = args.dataset

    train_and_save_model(data_dir)
