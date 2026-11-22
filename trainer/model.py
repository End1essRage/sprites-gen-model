import torch
import torch.nn as nn

class PixelTransformer(nn.Module):
    def __init__(self, vocab_size=17, d_model=64, context_size=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_size = context_size  # Сколько предыдущих пикселей смотрим
        
        # Эмбеддинг для пикселей
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Проекция текстового эмбеддинга (384 -> d_model)
        self.text_proj = nn.Linear(384, d_model)
        
        # Слой для обработки контекста (Conv1d лучше Linear для последовательностей)
        # in_channels = d_model, out_channels = d_model, kernel_size = context_size
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=context_size, padding=0)
        
        # Финальный слой для предсказания цвета
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, pixel_tokens, text_embedding):
        # pixel_tokens: [batch, seq_len]
        # text_embedding: [batch, 384]
        
        batch_size, seq_len = pixel_tokens.shape
        
        # 1. Эмбеддинги: [batch, seq_len, d_model]
        x = self.embedding(pixel_tokens)
        
        # 2. Conv1d ожидает [batch, channels, seq_len]
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        
        # 3. Паддинг, если нужно
        if seq_len < self.context_size:
            pad_size = self.context_size - seq_len
            x = torch.nn.functional.pad(x, (pad_size, 0))
            
        # 4. Свертка: [batch, d_model, new_seq_len]
        x = self.conv(x) 
        
        # 5. Берем последний временной шаг: [batch, d_model]
        x = x[:, :, -1] 
        
        # 6. Текстовый контекст
        text_emb = self.text_proj(text_embedding)  # [batch, d_model]
        x = x + text_emb
        
        # 7. Предсказание: [batch, vocab_size]
        logits = self.head(x) 
        
        # Возвращаем [batch, vocab_size] (без unsqueeze!)
        return logits