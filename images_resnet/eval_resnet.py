import argparse
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path


# The function for parsing the user arguments.
def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate ResNet-18 model on test images')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file (*.pt)')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the directory containing test images')
    return parser.parse_args()

# The class for providing the dataset for the testing part.
class TestingDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]
        
        # The normalization values are taken from the following source:
        # Source: https://pytorch.org/vision/0.15/transforms.html#transforms-scriptability
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

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

def extract_filename_without_extension(file_path):
    path = Path(file_path)
    return path.stem

# The main function for the model evaluation.
def evaluate_model(model_path, dataset_path):
    model = ResNet18()
    model.load_state_dict(torch.load(model_path))
    model.eval()    

    # Create custom dataset for test images.
    test_dataset = TestingDataset(dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    # Run inference.
    predictions = []

    with torch.no_grad():
        for images, paths in test_loader:            
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            probability = probabilities.item()
            predicted_label = '1' if probability > 0.5 else '0'
            predictions.append((paths[0], probability, predicted_label))

    # Sort the list based on the first element of each tuple.
    sorted_predictions = sorted(predictions, key=lambda x: x[0])

    # Print predictions.
    for img_path, probability, label in sorted_predictions:
        img_name = extract_filename_without_extension(img_path)
        print(f'{img_name} {probability} {label}')

if __name__ == '__main__':
    args = parse_arguments()
    evaluate_model(args.model, args.dataset)
