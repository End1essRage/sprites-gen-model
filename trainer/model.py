import torch
import torch.nn as nn

class PixelTransformer(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        
        self.text_proj = nn.Linear(384, d_model)
        
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.bn1 = nn.BatchNorm1d(d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model * 8)
        self.bn2 = nn.BatchNorm1d(d_model * 8)
        
        # Выход: 256 пикселей * 3 канала (RGB)
        self.fc3 = nn.Linear(d_model * 8, 256 * 3) 
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() # Ограничиваем цвета от 0 до 1

    def forward(self, text_embedding):
        t = self.text_proj(text_embedding)
        
        x = self.relu(self.bn1(self.fc1(t)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x) 
        
        # Форма: [batch, 256, 3]
        x = self.sigmoid(x)
        return x.view(-1, 256, 3)