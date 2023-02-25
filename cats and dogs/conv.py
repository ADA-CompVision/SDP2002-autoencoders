import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Define device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set the root directory for your dataset
data_dir = 'data'

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.enc_conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.enc_relu_1 = nn.ReLU()
        self.enc_conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc_relu_2 = nn.ReLU()
        self.enc_conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_relu_3 = nn.ReLU()
        self.enc_conv_4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_relu_4 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, 512)
        
        # Decoder
        self.dec_fc = nn.Linear(512, 2048)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 4, 4))
        self.dec_conv_1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_relu_1 = nn.ReLU()
        self.dec_conv_2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec_relu_2 = nn.ReLU()
        self.dec_conv_3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.dec_relu_3 = nn.ReLU()
        self.dec_conv_4 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)
        self.dec_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = self.enc_conv_1(x)
        x = self.enc_relu_1(x)
        x = self.enc_conv_2(x)
        x = self.enc_relu_2(x)
        x = self.enc_conv_3(x)
        x = self.enc_relu_3(x)
        x = self.enc_conv_4(x)
        x = self.enc_relu_4(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        # Decoder
        x = self.dec_fc(x)
        x = self.unflatten(x)
        x = self.dec_conv_1(x)
        x = self.dec_relu_1(x)
        x = self.dec_conv_2(x)
        x = self.dec_relu_2(x)
        x = self.dec_conv_3(x)
        x = self.dec_relu_3(x)
        x = self.dec_conv_4(x)
        x = self.dec_sigmoid(x)
        return x

# Define hyperparameters
batch_size = 64
learning_rate = 1e-3
num_epochs = 10

# Create a dataset and dataloader
# Prepare the data
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the autoencoder model
model = ConvAutoencoder()
# Print the model architecture and number of trainable parameters
print(model)
print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the autoencoder model
for epoch in range(num_epochs):
    running_loss = 0.0
    for data in dataloader:
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass 
        images = data
        with torch.no_grad():
    # Pass the batch of images directly to the model
           reconstructed_images = model(images)
        loss = criterion(reconstructed_images, images)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
# Save the model weights
#torch.save(model.state_dict(), 'autoencoder.pth')
