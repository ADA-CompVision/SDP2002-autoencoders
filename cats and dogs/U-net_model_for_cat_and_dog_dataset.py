import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor
import cv2
from sklearn.model_selection import train_test_split



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)
    
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_size=512):
        super().__init__()
        
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.bottom = DoubleConv(512, latent_size)

        self.up1 = nn.ConvTranspose2d(latent_size, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        self.conv4 = DoubleConv(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Downward path
        x1 = self.down1(x)
        x2 = F.max_pool2d(x1, kernel_size=2, stride=2)
        x3 = self.down2(x2)
        x4 = F.max_pool2d(x3, kernel_size=2, stride=2)
        x5 = self.down3(x4)
        x6 = F.max_pool2d(x5, kernel_size=2, stride=2)
        x7 = self.down4(x6)
        x8 = F.max_pool2d(x7, kernel_size=2, stride=2)
        x9 = self.bottom(x8)
        
        # Upward path
        x = self.up1(x9)
        x = torch.cat([x, x7], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x5], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        
        out = self.out(x)
        return out

# Define custom dataset for loading original images and masks
class MyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        image = Image.open(self.image_dir[idx]).convert('RGB')
        mask = Image.open(self.mask_dir[idx]).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask

# Define transformations for data preprocessing
transform = ToTensor()

# Define paths to original images and masks
image_dir = '/home/sign-lang/Desktop/sdp(Nihad)/da/images' # List of file paths to original images
mask_dir =  '/home/sign-lang/Desktop/sdp(Nihad)/da/m' # List of file paths to corresponding masks

# Create custom dataset
dataset = MyDataset(image_dir, mask_dir, transform=transform)

# Split the dataset into training and validation sets (e.g., 80%/20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation sets
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the UNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model = UNet(in_channels=3, out_channels=1, latent_size=512).to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate the model on the validation set
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
    
    val_loss /= len(val_dataset)
    
    # Print training loss and validation loss for this epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'unet_model.pth')


#------------------------------------------------------------------------------------------------------




import torch
import torchvision.transforms as transforms
from PIL import Image

# Define the path to your saved model


import torch

# Define the path to your model checkpoint file
model_path = '/home/sign-lang/Desktop/sdp(Nihad)/model.pth'

# Define the path to your test image
image_path = '/home/sign-lang/Desktop/sdp(Nihad)/DATA/images/1.jpg'

# Create an instance of your model
model = UNet()  # Replace MyModelClass with your actual model class

# Save the state dictionary during training
torch.save({'state_dict': model.state_dict()}, model_path)

# Load the state dictionary during testing
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])

# Set the model to evaluation mode
model.eval()

# Print the parameters of the model
for i, param in enumerate(model.parameters()):
    print(f"Parameter {i}: {param}")

# Load the test image
image = Image.open(image_path)

# Apply the same transformations as during training (e.g., resize, normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the same size as during training
    transforms.ToTensor(),           # Convert to a PyTorch tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Normalization values used during training
        std=[0.229, 0.224, 0.225]
    )
])

# Apply the transformations to the test image
image = transform(image)

# Add a batch dimension to the image tensor
image = image.unsqueeze(0)

# Pass the image through the model to get the predicted output
output = model(image)

# Get the predicted class index with the highest probability
#predicted_class_idx = torch.argmax(output, dim=1).item()

# Print the predicted class index
print('Predicted class index:', output)

# Load and preprocess the test image
test_image_path = '/home/sign-lang/Desktop/sdp(Nihad)/da/images/1.jpg'
test_image = cv2.imread(test_image_path)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
test_image = cv2.resize(test_image, (512, 512))
test_image = test_image.astype(np.float32) / 255.0
test_image_tensor = ToTensor()(test_image)
test_image_tensor = test_image_tensor.unsqueeze(0)

# Generate the output image from the test image using the trained model
with torch.no_grad():
    output_image_tensor = model(test_image_tensor)
output_image = output_image_tensor.squeeze(0).detach().cpu().numpy()
output_image = np.transpose(output_image, (1, 2, 0))
output_image = np.clip(output_image, 0, 1) * 255
output_image = output_image.astype(np.uint8)

# Convert numpy array to PIL Image
output_image = Image.fromarray(output_image)

# Save the generated output image
output_image.save('generated_output.jpg')