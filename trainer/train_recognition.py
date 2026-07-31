# train_recognition.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

# Character set definition (Arabic letters, numbers, spaces, and punctuation)
ALPHABET = (
    "-" # blank token for CTC loss
    " "
    "ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئةى"
    "٠١٢٣٤٥٦٧٨٩0123456789/-"
)

CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALPHABET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(ALPHABET)}
NUM_CLASSES = len(ALPHABET)

class ArabicOCRDataset(Dataset):
    def __init__(self, labels_file, img_dir, target_w=256, target_h=64):
        self.img_dir = img_dir
        self.target_w = target_w
        self.target_h = target_h
        self.samples = []
        
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 2:
                    self.samples.append((parts[0], parts[1]))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback to white image
            img = np.ones((self.target_h, self.target_w), dtype=np.uint8) * 255
            
        # Resize to standard size
        img = cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
        
        # Normalize to [-1, 1]
        img = (img.astype(np.float32) / 127.5) - 1.0
        img = np.expand_dims(img, axis=0) # Add channel dim: (1, H, W)
        
        # Encode label text into index sequence
        encoded = [CHAR_TO_IDX.get(char, 1) for char in label] # default to space if unknown
        
        return torch.tensor(img, dtype=torch.float32), torch.tensor(encoded, dtype=torch.long)

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_flat = torch.cat(targets)
    
    return images, targets_flat, target_lengths

# CRNN Architecture: CNN feature extractor + Bidirectional LSTM sequential layers
class CRNN(nn.Module):
    def __init__(self, img_h=64, num_classes=NUM_CLASSES, hidden_size=256):
        super(CRNN, self).__init__()
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # (64, H/2, W/2)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # (128, H/4, W/4)
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # pool H only: (256, H/8, W/4)
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # pool H only: (512, H/16, W/4)
            
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True) # (512, 1, W/4 - 1)
        )
        
        # RNN sequence model layers
        self.rnn = nn.Sequential(
            nn.LSTM(512, hidden_size, bidirectional=True, num_layers=2, batch_first=True, dropout=0.2)
        )
        
        # Linear decoder class mapping
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # 1. Feature extraction
        features = self.cnn(x) # Shape: [N, C, H, W] -> (N, 512, H_out, seq_len)
        features = features.mean(dim=2) # Collapse H dimension by averaging: Shape: [N, C, seq_len]
        features = features.permute(0, 2, 1) # Shape: [N, seq_len, C]
        
        # 2. Sequence modeling
        rnn_out, _ = self.rnn(features) # Shape: [N, seq_len, hidden * 2]
        
        # 3. Class probability projection
        out = self.fc(rnn_out) # Shape: [N, seq_len, num_classes]
        
        # CTC Loss requires shape [seq_len, N, num_classes]
        return out.permute(1, 0, 2)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Check if dataset exists, generate it if not
    if not os.path.exists("dataset/labels.txt"):
        print("Dataset not found. Generating synthetic dataset first...")
        os.system("python3 generate_synthetic_data.py")
        
    dataset = ArabicOCRDataset("dataset/labels.txt", "dataset/images")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    model = CRNN().to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    epochs = 15
    print(f"Starting fine-tuning for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (images, targets, target_lengths) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # Output shape: [seq_len, batch_size, num_classes]
            outputs = model(images)
            
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long, device=device)
            
            # Calculate CTC Loss
            loss = criterion(outputs.log_softmax(2), targets, input_lengths, target_lengths)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {epoch_loss / len(dataloader):.4f}")
        
    # Save the custom weights
    torch.save(model.state_dict(), "custom_arabic_recognition.pth")
    print("Model trained and saved to custom_arabic_recognition.pth!")

if __name__ == "__main__":
    train()
