import torch
import torch.nn as nn
from torch.optim import AdamW
from model import PixelTransformer
from dataset import get_dataloader

model = PixelTransformer()
optimizer = AdamW(model.parameters(), lr=1e-3)

# MSE Loss для регрессии цветов
criterion = nn.MSELoss()

dataloader = get_dataloader(batch_size=16)

model.train()
for epoch in range(5):
    for step, batch in enumerate(dataloader):
        pixel_colors = batch["pixel_colors"] # [batch, 256, 3]
        text = batch["text_embedding"] 

        optimizer.zero_grad()
        
        logits = model(text) # [batch, 256, 3]
        
        # Считаем ошибку между предсказанными и реальными цветами
        loss = criterion(logits, pixel_colors)
        
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss (MSE): {loss.item():.4f}")

torch.save(model.state_dict(), "pixel_model_real.pth")