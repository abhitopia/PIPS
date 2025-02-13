import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from experiments.adaptive_weights.adaptive_weights import AdaptiveWeightsAdjustor
# ====================================================
# MNIST Autoencoder Example with Adaptive Loss Coefficients
# ====================================================

# Define a simple classifier for MNIST.
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        # Encoder: simple MLP
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        # Classifier: output layer for 10 classes
        self.classifier = nn.Linear(32, 10)
        
    def forward(self, x):
        latent = self.encoder(x)
        logits = self.classifier(latent)
        return logits

# Define a function to compute weight decay loss (L2 norm) over all parameters.
def compute_weight_decay_loss(model):
    decay_loss = 0.0
    for param in model.parameters():
        decay_loss += torch.sum(param ** 2)
    return decay_loss

# Main training code for MNIST classifier.
def train_mnist_classifier(num_epochs=1, batch_size=64, adaptive_update_lag=10):
    # Use MNIST dataset, normalized to [0,1]
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./mnist_data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cpu")
    model = MNISTClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Create an adaptive manager for two metrics:
    #   "class_loss": classification loss; "wd": weight decay loss.
    # We initialize with, say, 0.8 for class_loss and 0.2 for wd.
    adaptive_manager = AdaptiveWeightsAdjustor(
        metric_names=["class_loss", "wd"],
        w_init={"class_loss": 0.99, "wd": 0.01},
        eta=0.1,
        alphaFuture=0.1,
        alphaPast=0.5,
        lag=adaptive_update_lag,
        max_weight=1.0,
        master_scale=1.0
    )
    
    criterion = nn.CrossEntropyLoss()  # classification loss criterion
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass: classify
            logits = model(data)
            class_loss = criterion(logits, target)
            
            # Compute weight decay loss manually
            wd_loss = compute_weight_decay_loss(model)
            
            # Get adaptive coefficients for each loss
            w_class_loss = adaptive_manager.get_weight("class_loss")
            w_wd = adaptive_manager.get_weight("wd")
            
            total_loss = w_class_loss * class_loss + w_wd * wd_loss
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            # Update the adaptive manager using the current loss values.
            adaptive_manager.step({"class_loss": class_loss.item(), "wd": wd_loss.item()})
            
            # Optionally, print every 100 batches.
            if batch_idx % 100 == 0 or True:
                print(f"Epoch [{epoch+1}], Batch [{batch_idx}], Total Loss: {total_loss.item():.4f}, "
                      f"Class Loss: {class_loss.item():.4f}, WD: {wd_loss.item():.4e}")
                print("Adaptive Weights:", adaptive_manager.get_all_weights())
                
        print(f"Epoch [{epoch+1}] Average Loss: {running_loss/len(train_loader):.4f}")
    
    print("Training complete. Final adaptive weights:", adaptive_manager.get_all_weights())

# ====================================================
# Run the training function
# ====================================================
if __name__ == "__main__":
    train_mnist_classifier(num_epochs=10, batch_size=64, adaptive_update_lag=5)
