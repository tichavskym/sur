import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
import os
import argparse

class PersonDetectionDatasetTest(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Define ResNet18 model
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.resnet(x)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate ResNet-18 model on test images')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the directory containing test images')
    return parser.parse_args()

def extract_filename_without_extension(file_path):
    path = Path(file_path)
    return path.stem

def evaluate_model(model_path, dataset_path):
    # Load the trained model
    model = ResNet18()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define transformations for preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create custom dataset for test images
    test_dataset = PersonDetectionDatasetTest(dataset_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Perform inference on test images
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions = []

    with torch.no_grad():
        for images, paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            probability = probabilities.item()
            predicted_label = '1' if probability > 0.5 else '0'
            predictions.append((paths[0], probability, predicted_label))

    # Print predictions
    for img_path, probability, label in predictions:
        img_name = extract_filename_without_extension(img_path)
        print(f'{img_name} {probability} {label}')

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments()

    # Evaluate the model on test images
    evaluate_model(args.model, args.dataset)
