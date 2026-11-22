import torch
import torch.nn as nn
from torch.optim import AdamW
from model import PixelTransformer
from dataset import get_dataloader

# 1. Инициализация
model = PixelTransformer(vocab_size=17) # 16 цветов + 1 прозрачный (индекс 0)
optimizer = AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 2. Загрузка реальных данных
dataloader = get_dataloader(batch_size=16)

# 3. Обучение
model.train()
for epoch in range(1):
    for step, batch in enumerate(dataloader):
        pixels = batch["pixel_tokens"] # [batch, 256]
        text = batch["text_embedding"] # [batch, 384]
        
        # Мы будем учить модель на случайных позициях в последовательности
        # Чтобы она видела разные контексты
        batch_size = pixels.size(0)
        loss_total = 0
        
        # Для простоты учим на 10 случайных шагах за батч
        for _ in range(10):
           # Выбираем случайную позицию
            pos = torch.randint(16, 256, (1,)).item()
            
            # Вход: пиксели с 0 по pos-1
            input_pixels = pixels[:, :pos]
            
            # Цель: пиксель на позиции pos. 
            # ВАЖНО: приводим к long и убеждаемся, что форма [batch]
            target_pixel = pixels[:, pos].long().view(-1) 
            
            optimizer.zero_grad()
            
            logits = model(input_pixels, text) 
            # logits уже имеет форму [batch, vocab_size]
            
            # Считаем лосс
            loss = criterion(logits, target_pixel)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss_total / 10:.4f}")
            
        if step >= 50:
            break

# 4. Сохранение
torch.save(model.state_dict(), "pixel_model_real.pth")
print("Модель обучена на реальных данных и сохранена.")